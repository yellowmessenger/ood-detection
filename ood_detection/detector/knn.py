import pandas as pd
import numpy as np
from ood_detection.classifier.feature_extractor import load_feature_extractor, build_features
from ood_detection.detector.base import BaseDetector

class KNN(BaseDetector):
    def __init__(self,feature_extractor: str, ood_label: str) -> None:
        BaseDetector.__init__(self) 
        self.feature_extractor = feature_extractor
        self.ood_label = ood_label
        
        # This parameter will be used to decide the prediction class
        # If True, the lower the score, the more likely it's outdomain
        # Else, the higher the score, the more likely it's outdomain
        self.outdomain_is_lower = False 

    def fit(self,df: pd.DataFrame):
        if self.ood_label in df['intent'].unique():
            print(f"Found {self.ood_label} in training data. This detector can only be used when Out-Domain data does not exist in the training data.")
            return

        # Generate Embeddings
        x_train,_= build_features(self.feature_extractor,
                                  df['text'],df['intent'],
                                  model=load_feature_extractor(self.feature_extractor))
        self.train_embeddings = x_train
        
    def predict_score(self,df_test: pd.DataFrame):
        q_embeddings,_ = build_features(self.feature_extractor,
                                  df_test['text'],df_test['text'],
                                  model=load_feature_extractor(self.feature_extractor))
        
        euclid_dist = np.array([np.linalg.norm(self.train_embeddings-q_emb,axis=1) for q_emb in q_embeddings])
        return np.min(euclid_dist, axis=1)  
