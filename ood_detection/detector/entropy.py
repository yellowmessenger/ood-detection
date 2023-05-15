import pandas as pd
import numpy as np
from ood_detection.classifier.train import train_classifier
from ood_detection.classifier.feature_extractor import load_feature_extractor, build_features
from ood_detection.detector.base import BaseDetector

class Entropy(BaseDetector):
    def __init__(self,feature_extractor: str) -> None:
        BaseDetector.__init__(self) 
        self.feature_extractor = feature_extractor

        # This parameter will be used to decide the prediction class
        # If True, the lower the score, the more likely it's outdomain
        # Else, the higher the score, the more likely it's outdomain
        self.outdomain_is_lower = False 

    def fit(self,df: pd.DataFrame, use_best_ckpt: bool = False):
        # Fit Classifier
        model_name = "mlp" if not use_best_ckpt else "mlp_best_ckpt"
        clf = train_classifier(df, model_name, self.feature_extractor, skip_cv = True)

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
