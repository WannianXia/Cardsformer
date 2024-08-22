from typing import OrderedDict
import torch
from torch.utils.data import DataLoader
from PredictionDataset import PredictionDataset
from tqdm import tqdm
from Model.PredictionModel import PredictionModel
import torch.optim
from Algo.utils import log
from tensorboardX import SummaryWriter
from tqdm import tqdm
import numpy as np
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6"

train_step = 100000

model = PredictionModel(is_train=True)


data = PredictionDataset([i for i in range(10)])
data_test = PredictionDataset([0, ], True)

data_train = DataLoader(data, batch_size=20000, shuffle=True, num_workers=0)
data_test = DataLoader(data_test, batch_size=5000, shuffle=True)


device = 'cpu'
if torch.cuda.is_available():
    device = torch.cuda.current_device()
    model = torch.nn.DataParallel(model).to(device)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
best_test_loss = 100

writer = SummaryWriter('runs/prediction_model')

for epoch in range(train_step):
    log.info("Epoch {} / {}".format(epoch, train_step))
    losses = []
    for batch in tqdm(data_train, position=0, leave=True):
        optimizer.zero_grad()
        for i in range(8):
            batch[i] = batch[i].float().to(device)
        x = [batch[i] for i in range(6)]
        y = [batch[6], batch[7]]
        pred = model(x)
        loss1 = loss_fn(y[0], pred[0])
        loss2 = loss_fn(y[1], pred[1])
        loss = loss1 + loss2
        losses.append(loss.mean().item())
        loss.backward()
        optimizer.step()
    writer.add_scalar('training_loss', np.mean(losses), epoch)
    log.info("Current Training Loss is: {}".format(loss))
    test_losses = []
    for batch in data_test:
        for i in range(3, 8):
            batch[i] = batch[i].float().to(device)
        x = [batch[i] for i in range(6)]
        y = [batch[6], batch[7]]
        with torch.no_grad():
            pred = model(x)
        loss1 = loss_fn(y[0], pred[0])
        loss2 = loss_fn(y[1], pred[1])
        loss = loss1 + loss2
        test_losses.append(loss.mean().item())
    test_loss = np.mean(test_losses)
    writer.add_scalar('test_loss', test_loss, epoch)
    if test_loss < best_test_loss:
        log.info('minion loss: {} \t hero loss: {}'.format(loss1, loss2))
        log.info('Best loss: {}'.format(test_loss))
        best_test_loss = test_loss
        torch.save(
            {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, 'trained_models/prediction_model' + str(epoch) + '.tar'
            )
    writer.add_scalar('best_test_loss', best_test_loss.item(), epoch)
