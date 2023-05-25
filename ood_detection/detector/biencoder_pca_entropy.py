import pandas as pd
import numpy as np
from ood_detection.classifier.train import train_classifier
from ood_detection.detector.base import BaseDetector
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

class BiEncoderPCAEntropy(BaseDetector):
    def __init__(self,feature_extractor: str, ood_label: str) -> None:
        BaseDetector.__init__(self) 
        self.feature_extractor = feature_extractor
        self.ood_label = ood_label

        if self.feature_extractor not in ['mpnet']:
            raise NotImplementedError("Currently only 'mpnet' is supported. You can add any new sentence-trasnformer model.")

        # This parameter will be used to decide the prediction class
        # If True, the lower the score, the more likely it's outdomain
        # Else, the higher the score, the more likely it's outdomain
        self.outdomain_is_lower = False

    def fit(self,df: pd.DataFrame):  
        if self.ood_label in df['intent'].unique():
            print(f"Found {self.ood_label} in training data. This detector can only be used when Out-Domain data does not exist in the training data.")
            return
      
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
        sims = cosine_similarity(q_pca_transformed, self.train_pca_transformed)
        probas_adj = np.array([softmax_adj(sim) for sim in sims])
        entropy_score = calc_entropy(probas_adj)
        return entropy_score


def calc_entropy(probas):
  return np.sum(probas * np.log(probas + 1e-6),axis=1) * -1


def softmax_adj(prob):
  top_idx,top_score = np.argmax(prob),np.max(prob)
  
  excluded_prob = np.delete(prob,top_idx)
  excluded_prob = (1-top_score) * (np.exp(excluded_prob)/np.sum(np.exp(excluded_prob)))
  return np.insert(excluded_prob,top_idx,top_score)