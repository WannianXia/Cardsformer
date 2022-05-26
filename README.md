# Cardsformer
This repository is the official implementation of Cardsformer: Grouding Language to Learn a Generalized Policy in Hearthstone.

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
