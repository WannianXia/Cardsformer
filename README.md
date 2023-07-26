# Cardsformer
This repository is the official implementation of *Cardsformer: Grouding Language to Learn a Generalizable Policy in Hearthstone*.

![Cardsformer Model Architecture](https://github.com/WannianXia/Cardsformer/blob/main/imgs/Cardsformer_Model.png?raw=true)

## Requirements
Python Packages:
- pytorch
- pythonnet==2.5.2
- numpy
- pandas
- tensorboardx
- tqdm
- transformers

Others:
- [Mono](https://www.mono-project.com)

## Training
To prepare offline trajectories to train the Prediction Model, run:
```
python gen_data.py
```
With collected data, to train the Prediction Model, run:
```
python train_prediction.py
```
To train the Policy Model, run:
```
python train_policy.py
```

## Evaluation
Baseline models of Hearthstone AI Competetion are available in [here](https://hearthstoneai.github.io/botdownloads.html). You can construct DLL files using these baseline method classes and refer them in python code. 

You can download the pretrained models and run ```eval.py``` (modify the model path before you do) to record a game log and see how the agent performs.

## Pretrained Models
Pretrained models of Cardsformer are available at [Google Drive](https://drive.google.com/drive/folders/1ZY1LhhLpBr0GzcM1GYOoaiMm1pmoAduC?usp=sharing).
The Policy Models are original model checkpoints, the Prediction Model is a saved dict and should be unwrapped by following:
```
prediction_model = PredictionModel()

checkpoint_states = torch.load(model_path, map_location=device)['model_state_dict']

new_state_dict = typing.OrderedDict()
for k, v in checkpoint_states.items():
    name = k[7:]
    new_state_dict[name] = v

prediction_model.load_state_dict(new_state_dict)
prediction_model.eval()
```
