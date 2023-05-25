# OOD-Detection

...

## Introduction

...


## Quick Start
1. Use anaconda to create Python (version >= 3.8) environment with Poetry
```
conda create --name ood_detection python=3.8 poetry=1.4.0
conda activate textoir
```
2. Clone this repository
```
git clone https://github.com/yellowmessenger/ood-detection.git
cd ood-detection
```
3. Install requirements with `poetry`
```
poetry install  
```

4. (Optional) if you want to run LikelihoodRatio, you also need to download the Glove 6B 100D embeddings. You can download it from [here](https://www.kaggle.com/datasets/danielwillgeorge/glove6b100dtxt). Please make sure to put it in the `ood_detection/dataloaders/raw` folder.

## Citations

If this work is helpful, or you want to use the codes and results in this repo, please cite our paper:

```

```


## Contributors

[Louis Owen](https://github.com/louisowen6), 

## Bugs or questions?

If you have any questions, please open issues and illustrate your problems as detailed as possible. If you want to integrate your method in our repo, please feel free to **pull request**!