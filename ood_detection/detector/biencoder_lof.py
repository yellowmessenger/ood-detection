import pandas as pd
import numpy as np
from ood_detection.classifier.train import train_classifier
from sklearn.neighbors import LocalOutlierFactor
from ood_detection.detector.base import BaseDetector

class BiEncoderLOF(BaseDetector):
    def __init__(self,feature_extractor: str) -> None:
        BaseDetector.__init__(self) 
        self.feature_extractor = feature_extractor

        if self.feature_extractor not in ['mpnet']:
            raise NotImplementedError("Currently only 'mpnet' is supported. You can add any new sentence-trasnformer model.")

        # This parameter will be used to decide the prediction class
        # If True, the lower the score, the more likely it's outdomain
        # Else, the higher the score, the more likely it's outdomain
        self.outdomain_is_lower = True

        print("="*50)
        print("This Detector can only be used when Out-Domain data does not exist in the training data.")
        print("="*50)

    def fit(self,df: pd.DataFrame):   
        if self.ood_label in df['intent'].unique():
            print(f"Found {self.ood_label} in training data. This detector can only be used when Out-Domain data does not exist in the training data.")
            return
     
        # Fit Classifier
        model_name = "biencoder"
        clf = train_classifier(df, model_name, self.feature_extractor, skip_cv = True)

        self.clf = clf

        x_train = self.clf.clf.encode(df['text'].to_list())
        lof = LocalOutlierFactor(n_neighbors=10, novelty = True, metric = 'cosine')
        lof.fit(x_train)
        
        self.lof = lof

    def predict_score(self,df_test: pd.DataFrame):
        if 'text' not in df_test.columns:
            print("column 'text' is missing in df_test.")
            return
        
        x_test = self.clf.clf.encode(df_test['text'].to_list())
        lof_score = self.lof.decision_function(x_test)
        
        return lof_score
