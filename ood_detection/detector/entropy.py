import pandas as pd
import numpy as np
from ood_detection.classifier.train import train_classifier
from ood_detection.classifier.feature_extractor import load_feature_extractor, build_features
from ood_detection.detector.base import BaseDetector

class Entropy(BaseDetector):
    def __init__(self,feature_extractor: str, ood_label: str) -> None:
        BaseDetector.__init__(self) 
        self.feature_extractor = feature_extractor
        self.ood_label = ood_label

        # This parameter will be used to decide the prediction class
        # If True, the lower the score, the more likely it's outdomain
        # Else, the higher the score, the more likely it's outdomain
        self.outdomain_is_lower = False 

        print("="*50)
        print("This Detector can only be used when Out-Domain data does not exist in the training data.")
        print("="*50)

    def fit(self,df: pd.DataFrame, use_best_ckpt: bool = False,
            df_val_ckpt: pd.DataFrame = None):
        if self.ood_label in df['intent'].unique():
            print(f"Found {self.ood_label} in training data. This detector can only be used when Out-Domain data does not exist in the training data.")
            return
        
        # Fit Classifier
        model_name = "mlp_best_ckpt" if use_best_ckpt else "mlp"
        clf = train_classifier(df, model_name, self.feature_extractor,
                               df_val_ckpt =  df_val_ckpt,
                               skip_cv = True)

        self.clf = clf

    def predict_score(self,df_test: pd.DataFrame):
        x_test,_ = build_features(self.feature_extractor,
                                  df_test['text'],df_test['text'],
                                  model=load_feature_extractor(self.feature_extractor))
        probas = self.clf.predict_proba(x_test)

        # Compute entropy score
        entropy_score = calc_entropy(probas)

        return entropy_score


def calc_entropy(probas):
  return np.sum(probas * np.log(probas + 1e-6),axis=1) * -1
