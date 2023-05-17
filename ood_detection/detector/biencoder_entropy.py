import pandas as pd
import numpy as np
from ood_detection.classifier.train import train_classifier
from ood_detection.detector.base import BaseDetector
from sklearn.metrics.pairwise import cosine_similarity

class BiEncoderEntropy(BaseDetector):
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
        self.train_embeddings = self.clf.clf.encode(df['text'].to_list())

    def predict_score(self,df_test: pd.DataFrame):
        if 'text' not in df_test.columns:
            print("column 'text' is missing in df_test.")
            return
        
        q_embeddings = self.clf.clf.encode(df_test['text'].to_list())
        sims = cosine_similarity(q_embeddings, self.train_embeddings)
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