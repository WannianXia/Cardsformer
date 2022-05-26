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
`
python gen_data.py
`
