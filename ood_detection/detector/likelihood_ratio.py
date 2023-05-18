import pandas as pd
from sklearn.metrics import fbeta_score,matthews_corrcoef
from functools import partial
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping

from ood_detection.detector.Maha_OOD.lib.datasets.datasets import get_dataset_simple, collate_fn_simple as collate_fn
from ood_detection.detector.Maha_OOD.lib.modules.likelihood_ratio import LikelihoodratioModule
from ood_detection.detector.Maha_OOD.lib.utils import get_device_from_config
from ood_detection.detector.Maha_OOD.lib.data_utils import Vocab

import os
dirname = os.path.dirname(os.path.dirname(__file__))

import yaml
from munch import DefaultMunch

class LikelihoodRatio():
    def __init__(self,config_name: str) -> None:
        if config_name in ['clinc150','clinc']:
            config_name = 'clinc'
        if config_name.startswith('snips'):
            config_name = 'snips'

        with open(f'{dirname}/config_likelihood_{config_name}.yaml', 'r') as stream:
            config = yaml.safe_load(stream)
        print(config)
        config = DefaultMunch.fromDict(config)
        # device = get_device_from_config(config=config)

        training_config = config.training
        datasets, vocab = get_dataset_simple(config.dataset.name,
                                            add_valid_to_vocab=config.add_valid_to_vocab,
                                            add_test_to_vocab=config.add_test_to_vocab,
                                            **config.dataset
                                            )

        self.module = LikelihoodratioModule(config, vocab=vocab,
                                    collate_fn=partial(collate_fn,
                                                        pad_idx=vocab.pad_idx,
                                                        bos_idx=vocab.bos_idx,
                                                        eos_idx=vocab.eos_idx
                                                        ),
                                    train_dataset=datasets['train'],
                                    val_dataset=datasets['val'],
                                    test_dataset=datasets['test'])

        self.trainer = Trainer(
            # gpus=device,
            early_stop_callback=EarlyStopping(**training_config.early_stop),
            max_epochs=training_config.max_epochs,
            gradient_clip_val=training_config.get('gradient_clip_val', 0),
            checkpoint_callback=False
        )

    def fit(self):
        self.trainer.fit(self.module)

    def predict(self):
        raise NotImplementedError("LikelihoodRatio not support prediction for now.")
    
    def predict_score(self):
        raise NotImplementedError("LikelihoodRatio not support prediction for now.")

    def tune_threshold(self):
        raise NotImplementedError("LikelihoodRatio not support threshold tuning for now.")
    
    def benchmark(self):
        self.trainer.test()
