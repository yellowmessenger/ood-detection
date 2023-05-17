import pandas as pd
import numpy as np
from ood_detection.classifier.train import train_classifier
from ood_detection.detector.base import BaseDetector
from sklearn.decomposition import PCA

class BiEncoderPCAEuclidean(BaseDetector):
    def __init__(self,feature_extractor: str) -> None:
        BaseDetector.__init__(self) 
        self.feature_extractor = feature_extractor

        if self.feature_extractor not in ['mpnet']:
            raise NotImplementedError("Currently only 'mpnet' is supported. You can add any new sentence-trasnformer model.")

        # This parameter will be used to decide the prediction class
        # If True, the lower the score, the more likely it's outdomain
        # Else, the higher the score, the more likely it's outdomain
        self.outdomain_is_lower = False

        print("="*50)
        print("This Detector can only be used when Out-Domain data does not exist in the training data.")
        print("="*50)

    def fit(self,df: pd.DataFrame):        
        # Fit Classifier
        model_name = "biencoder"
        clf = train_classifier(df, model_name, self.feature_extractor, skip_cv = True)
        self.clf = clf
        train_embeddings = self.clf.clf.encode(df['text'].to_list())

        pca = PCA()
        self.train_pca_transformed = pca.fit_transform(train_embeddings)

        self.pca = pca

    def predict_score(self,df_test: pd.DataFrame):
        if 'text' not in df_test.columns:
            print("column 'text' is missing in df_test.")
            return
        
        q_embeddings = self.clf.clf.encode(df_test['text'].to_list())
        q_pca_transformed = self.pca.transform(q_embeddings)
        euclid_dist = np.array([np.linalg.norm(self.train_pca_transformed-q_emb,axis=1) for q_emb in q_pca_transformed])
        return np.min(euclid_dist, axis=1)  
