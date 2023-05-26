import pandas as pd
import numpy as np
from ood_detection.classifier.train import train_classifier
from ood_detection.classifier.feature_extractor import load_feature_extractor, build_features
from ood_detection.detector.base import BaseDetector

class BiEncoderCosine(BaseDetector):
    def __init__(self,feature_extractor: str, ood_label: str) -> None:
        BaseDetector.__init__(self) 
        self.feature_extractor = feature_extractor
        self.ood_label = ood_label

        if self.feature_extractor not in ['mpnet']:
            raise NotImplementedError("Currently only 'mpnet' is supported. You can add any new sentence-trasnformer model.")

        # This parameter will be used to decide the prediction class
        # If True, the lower the score, the more likely it's outdomain
        # Else, the higher the score, the more likely it's outdomain
        self.outdomain_is_lower = True

    def fit(self,df: pd.DataFrame):    
        if self.ood_label in df['intent'].unique():
            print(f"Found {self.ood_label} in training data. This detector can only be used when Out-Domain data does not exist in the training data.")
            return "error"
    
        # Fit Classifier
        model_name = "biencoder"
        clf = train_classifier(df, model_name, self.feature_extractor, skip_cv = True)

        self.clf = clf
        self.df_train = df

    def predict_score(self,df_test: pd.DataFrame):
        if 'text' not in df_test.columns:
            print("column 'text' is missing in df_test.")
            return
        
        probas = self.clf.predict_proba(df_test['text'].to_list(),
                                        self.df_train['text'].to_list()
                                        )
        return probas
