# BED

## Quick Start

1. Use anaconda to create Python (version = 3.8) environment with Poetry
```
conda create --name ood_detection python=3.8 poetry=1.4.0
conda activate ood_detection
```
2. Clone this repository
```
git clone https://github.com/yellowmessenger/ood-detection.git
cd ood-detection
```
3. Install requirements with `poetry`
```
poetry config virtualenvs.create false
poetry install  
```

4. (Optional) if you want to run LikelihoodRatio, you also need to download the Glove 6B 100D embeddings. You can download it from [here](https://www.kaggle.com/datasets/danielwillgeorge/glove6b100dtxt). Please make sure to put it in the `ood_detection/dataloaders/raw` folder.

5. Run the benchmarking script

```
sh run_benchmark.sh
```

6. Generate the benchmarking table

```
python get_benchmarking_table.py --results_dir ./benchmarking_results --output_dir .
```

## Usage

You can get the benchmarking results for a specific method via the `benhcmark.py` script.

```
usage: benchmark.py [-h] --output_dir OUTPUT_DIR --dataset {clinc150,rostd,snips} --detector
                    {TrustScores,Entropy,LOF,BinaryMSP,MSP,DOC,ADB,KNN,MCDropout,BNNVI,BiEncoderCosine,BiEncoderLOF,BiEncoderEuclidean,BiEncoderMaha,BiEncoderEntropy,BiEncoderPCAEntropy,BiEncoderPCACosine,BiEncoderPCAEuclidean,RAKE,LikelihoodRatio}
                    [--feature_extractor {mpnet,use,bert,mpnet_best_ckpt,use_best_ckpt,bert_best_ckpt}]
                    [--use_best_ckpt USE_BEST_CKPT] [--is_ood_label_in_train IS_OOD_LABEL_IN_TRAIN] --ood_label
                    OOD_LABEL [--adb_alpha ADB_ALPHA] [--adb_step_size ADB_STEP_SIZE]

optional arguments:
  -h, --help            show this help message and exit
  --output_dir OUTPUT_DIR
                        The output directory where all benchmarking results will be written.
  --dataset {clinc150,rostd,snips}
                        The name of the dataset to train selected
  --detector {TrustScores,Entropy,LOF,BinaryMSP,MSP,DOC,ADB,KNN,MCDropout,BNNVI,BiEncoderCosine,BiEncoderLOF,BiEncoderEuclidean,BiEncoderMaha,BiEncoderEntropy,BiEncoderPCAEntropy,BiEncoderPCACosine,BiEncoderPCAEuclidean,RAKE,LikelihoodRatio}
                        which detector to use
  --feature_extractor {mpnet,use,bert,mpnet_best_ckpt,use_best_ckpt,bert_best_ckpt}
                        which feature extractor to use
  --use_best_ckpt USE_BEST_CKPT
                        whether to use best checkpoint of the classifier based on validation data
  --is_ood_label_in_train IS_OOD_LABEL_IN_TRAIN
                        whether to add ood label in the training data
  --ood_label OOD_LABEL
                        name of the ood label
  --adb_alpha ADB_ALPHA
                        alpha hyperparameter for ADB detector
  --adb_step_size ADB_STEP_SIZE
                        step_size hyperparameter for ADB detector
```

This project is written in a modular fashion. You can also import a specific dataset, method, and do anything that you want.

```python
from ood_detection import DataLoader,MSP

loader = DataLoader()
data = loader.load('clinc150')

det = MSP(feature_extractor='use',is_ood_label_in_train=True,ood_label='oos')
det.fit(data['train'])
pred_scores = det.predict_score(data['test'])

best_threshold = det.tune_threshold(data['val'],'oos',False) #if set to True, you'll get viz of different possible thresholds value
pred_cls = det.predict(data['test'],best_threshold)

benchmark_dict = det.benchmark(data['test'],data['val'],'oos')
```

If you want to use custom dataset, you can add your own dataset in the `ood_detection/dataloaders/base.py` and put the data under the `ood_detection/dataloaders/raw` folder. Just make sure your custom data has `text` and `intent` in the column names.

## Citations

If this work is helpful, or you want to use the codes and results in this repo, please cite our paper:

```bibtex
@misc{owen2023bed,
    title={BED},
    author={Louis Owen and Biddwan Ahmed and Abhay Kumar},
    year={2023},
    eprint={2306.08852},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```


## Contributors

[Louis Owen](https://www.linkedin.com/in/louisowen/), [Biddwan Ahmed](https://www.linkedin.com/in/biddwan-ahmed-917333126/), [Abhay Kumar](https://www.linkedin.com/in/akanyaani/)

## Bugs or questions?

If you have any questions, please open issues and illustrate your problems as detailed as possible. If you want to integrate your method in our repo, please feel free to **pull request**!
