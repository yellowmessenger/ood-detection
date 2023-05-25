import pandas as pd
import numpy as np
from ood_detection.classifier.train import train_classifier
from ood_detection.classifier.feature_extractor import load_feature_extractor, build_features
from ood_detection.detector.base import BaseDetector

class BinaryMSP(BaseDetector):
    def __init__(self,feature_extractor: str, ood_label: str) -> None:
        BaseDetector.__init__(self) 
        self.feature_extractor = feature_extractor
        self.ood_label = ood_label

        # This parameter will be used to decide the prediction class
        # If True, the lower the score, the more likely it's outdomain
        # Else, the higher the score, the more likely it's outdomain
        self.outdomain_is_lower = False 

    def fit(self,df: pd.DataFrame, use_best_ckpt: bool = False, 
            df_val_ckpt: pd.DataFrame = None):        
        # Convert into binary classes
        df_binary = df[['text','intent']].copy()
        df_binary['intent'] = df_binary['intent'].apply(lambda x: x == ood_label)

        # Fit Classifier
        model_name = "mlp_best_ckpt" if use_best_ckpt else "mlp"
        clf = train_classifier(df_binary, model_name, self.feature_extractor, 
                               df_val_ckpt = df_val_ckpt,
                               skip_cv = True)

        self.clf = clf

    def predict_score(self,df_test: pd.DataFrame):
        x_test,_ = build_features(self.feature_extractor,
                                  df_test['text'],df_test['text'],
                                  model=load_feature_extractor(self.feature_extractor))
        probas = self.clf.predict_proba(x_test)
        probas = probas[:,self.clf.trained_classes_mapping.index(True)] #get out-domain class probas

        return probas
