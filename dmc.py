
import os
import threading
import time
import timeit  # 测试代码片段执行时间
import pprint  # pretty print 一种美化输出的工具
from collections import deque  # 双向队列
import numpy as np

import torch
from torch import multiprocessing as mp  # torch提供的多线程，如何使用？
from torch import nn


from Model.ModelWrapper import Model  # 网络模型
from Algo.utils import get_batch, log, create_buffers, create_optimizers, act  # 这几个函数是干什么用的？

mean_episode_return_buf = {
    p: deque(maxlen=100)
    for p in ['Player1', 'Player2']
}

def compute_loss(logits, targets):
    loss = ((logits.view(-1) - targets)**2).mean()
    return loss

def learn(position, actor_models, model, batch, optimizer, flags, lock):
    """Performs a learning (optimization) step."""
    if flags.training_device != "cpu":
        device = torch.device('cuda:' + str(flags.training_device))
    else:
        device = torch.device('cpu')
    hand_card_embed = batch['hand_card_embed'].to(device)
    minion_embed = batch['minion_embed'].to(device)
    weapon_embed = batch['weapon_embed'].to(device)
    secret_embed = batch['secret_embed'].to(device)
    hand_card_scalar = batch['hand_card_scalar'].to(device)
    minion_scalar = batch['minion_scalar'].to(device)
    hero_scalar = batch['hero_scalar'].to(device)
    target = torch.flatten(batch['target'].to(device), 0, 1)
    episode_returns = batch['episode_return'][batch['done']]
    mean_episode_return_buf[position].append(
        torch.mean(episode_returns).to(device))

    next_minion_scalar = batch['next_minion_scalar'].to(device)
    next_hero_scalar = batch['next_hero_scalar'].to(device)
    with lock:
        learner_outputs = model(hand_card_embed, minion_embed, secret_embed, weapon_embed, hand_card_scalar, minion_scalar, hero_scalar, next_minion_scalar, next_hero_scalar, num_options=None, actor = False)
        loss = compute_loss(learner_outputs, target)
        stats = {
            'mean_episode_return_' + position:
            torch.mean(
                torch.stack([_r for _r in mean_episode_return_buf[position]
                             ])).item(),
            'loss_' + position:
            loss.item(),
        }

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), flags.max_grad_norm)
        optimizer.step()

        for actor_model in actor_models.values():
            actor_model.get_model().load_state_dict(model.state_dict())
        return stats

def train(flags):
    """
    This is the main funtion for training. It will first
    initilize everything, such as buffers, optimizers, etc.
    Then it will start subprocesses as actors. Then, it will call
    learning function with  multiple threads.
    """
    if not flags.actor_device_cpu or flags.training_device != 'cpu':
        if not torch.cuda.is_available():
            raise AssertionError(
                "CUDA not available. If you have GPUs, please specify the ID after `--gpu_devices`. Otherwise, please train with CPU with `python3 train.py --actor_device_cpu --training_device cpu`"
            )
    checkpointpath = os.path.expanduser('%s/%s/%s' %
                           (flags.savedir, flags.xpid, 'model.tar'))

    T = flags.unroll_length
    B = flags.batch_size

    if flags.actor_device_cpu:
        device_iterator = ['cpu']
    else:
        device_iterator = range(flags.num_actor_devices)
        assert flags.num_actor_devices <= len(
            flags.gpu_devices.split(',')
        ), 'The number of actor devices can not exceed the number of available devices'

    models = {}
    for device in device_iterator:
        model = Model(device=device)
        model.share_memory()
        model.eval()
        models[device] = model
    buffers = create_buffers(flags, device_iterator)

    ctx = mp.get_context('spawn')
    free_queue = {}
    full_queue = {}
    for device in device_iterator:
        _free_queue = {
            'Player1': ctx.SimpleQueue(),
            'Player2': ctx.SimpleQueue()
        }
        _full_queue = {
            'Player1': ctx.SimpleQueue(),
            'Player2': ctx.SimpleQueue()
        }
        free_queue[device] = _free_queue
        full_queue[device] = _full_queue
    learner_model = Model(device=flags.training_device)


    optimizer = create_optimizers(flags, learner_model)
    stat_keys = [
        'mean_episode_return_Player1',
        'loss_Player1',
        'mean_episode_return_Player2',
        'loss_Player2',
    ]
    frames, stats = 0, {k: 0 for k in stat_keys}
    position_frames = {'Player1': 0, 'Player2': 0}

    # Load models if any
    if flags.load_model and os.path.exists(checkpointpath):
        checkpoint_states = torch.load(
            checkpointpath,
            map_location=("cuda:" + str(flags.training_device)
                          if flags.training_device != "cpu" else "cpu"))
        learner_model.get_model().load_state_dict(checkpoint_states["model_state_dict"])
        optimizer.load_state_dict(checkpoint_states["optimizer_state_dict"])
        for device in device_iterator:
            models[device].get_model().load_state_dict(learner_model.get_model().state_dict())
        stats = checkpoint_states["stats"]
        frames = checkpoint_states["frames"]
        position_frames = checkpoint_states["position_frames"]
        log.info(f"Resuming preempted job, current stats:\n{stats}")

    # Starting actor processes
    actor_processes = []
    for device in device_iterator:
        num_actors = flags.num_actors
        for i in range(num_actors):
            actor = ctx.Process(target=act,
                                args=(i, device, free_queue[device],
                                      full_queue[device], models[device],
                                      buffers[device], flags))
            actor.start()
            actor_processes.append(actor)

    def batch_and_learn(i,
                        device,
                        position,
                        local_lock,
                        position_lock,
                        lock=threading.Lock()):
        """Thread target for the learning process."""
        nonlocal frames, position_frames, stats
        while frames < flags.total_frames:
            batch = get_batch(free_queue[device][position],
                              full_queue[device][position],
                              buffers[device][position], flags, local_lock)
            _stats = learn(position, models, learner_model.get_model(), batch, optimizer, flags, position_lock)

            with lock:
                for k in _stats:
                    stats[k] = _stats[k]
                to_log = dict(frames=frames)
                to_log.update({k: stats[k] for k in stat_keys})
                frames += T * B
                position_frames[position] += T * B

    for device in device_iterator:
        for m in range(flags.num_buffers):
            free_queue[device]['Player1'].put(m)
            free_queue[device]['Player2'].put(m)

    threads = []
    locks = {}
    for device in device_iterator:
        locks[device] = {
            'Player1': threading.Lock(),
            'Player2': threading.Lock()
        }
    position_locks = {
        'Player1': threading.Lock(),
        'Player2': threading.Lock()
    }

    for device in device_iterator:
        for i in range(flags.num_threads):
            for position in ['Player1', 'Player2']:
                thread = threading.Thread(target=batch_and_learn,
                                          name='batch-and-learn-%d' % i,
                                          args=(i, device, position,
                                                locks[device][position],
                                                position_locks['Player1']))
                thread.start()
                threads.append(thread)

    def checkpoint(frames):
        if flags.disable_checkpoint:
            return
        log.info('Saving checkpoint to %s', checkpointpath)
        _model = learner_model.get_model()
        torch.save(
            {
                'model_state_dict': _model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                "stats": stats,
                'flags': vars(flags),
                'frames': frames,
                'position_frames': position_frames
            }, checkpointpath)

        # Save the weights for evaluation purpose
        model_weights_dir = os.path.expandvars(
            os.path.expanduser('%s/%s/%s' %
                                (flags.savedir, flags.xpid, 'Trained_weights_' + str(frames) + '.ckpt')))
        torch.save(
            learner_model.get_model().state_dict(),
            model_weights_dir)

    fps_log = []
    timer = timeit.default_timer
    try:
        last_save_frame = frames - (frames % flags.frame_interval)
        while frames < flags.total_frames:
            start_frames = frames
            position_start_frames = {
                k: position_frames[k]
                for k in position_frames
            }
            start_time = timer()
            time.sleep(5)

            if frames - last_save_frame > flags.frame_interval:
                checkpoint(frames - (frames % flags.frame_interval))
                last_save_frame = frames - (frames % flags.frame_interval)
            end_time = timer()
            fps = (frames - start_frames) / (end_time - start_time)
            fps_log.append(fps)
            if len(fps_log) > 24:
                fps_log = fps_log[1:]
            fps_avg = np.mean(fps_log)

            position_fps = {
                k: (position_frames[k] - position_start_frames[k]) /
                (end_time - start_time)
                for k in position_frames
            }
            log.info(
                'After %i (L:%i U:%i) frames: @ %.1f fps (avg@ %.1f fps) (L:%.1f U:%.1f) Stats:\n%s',
                frames,
                position_frames['Player1'],
                position_frames['Player2'], fps, fps_avg,
                position_fps['Player1'], position_fps['Player2'], pprint.pformat(stats))

    except KeyboardInterrupt:
        return
    else:
        for thread in threads:
            thread.join()
        log.info('Learning finished after %d frames.', frames)
