import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from ood_detection.classifier.train import train_classifier
from ood_detection.classifier.feature_extractor import load_feature_extractor, build_features
from ood_detection.detector.base import BaseDetector

class BNNVI(BaseDetector):
    def __init__(self,feature_extractor: str, is_ood_label_in_train: bool, 
                 ood_label: str = None) -> None:
        BaseDetector.__init__(self) 
        self.feature_extractor = feature_extractor

        # This parameter will be used during inference
        # to know which probability score should be used
        self.is_ood_label_in_train = is_ood_label_in_train
        self.ood_label = ood_label

        # This parameter will be used to decide the prediction class
        # If True, the lower the score, the more likely it's outdomain
        # Else, the higher the score, the more likely it's outdomain
        if is_ood_label_in_train:
            self.outdomain_is_lower = False 
        else:
            self.outdomain_is_lower = True     
        
    def fit(self,df: pd.DataFrame):
        if self.is_ood_label_in_train and self.ood_label is None:
            print("is_ood_label_in_train is set to True but ood_label is None. Make sure to give the value of ood_label column name.")
            return "error"         
        # Fit Classifier
        model_name = "mlp_dense_flipout"
        clf = train_classifier(df, model_name, self.feature_extractor, skip_cv = True,
                                n_additional_layers=1,n_units_l2=128,dropout_rate_l2=0.35
                                )

        self.clf = clf

    def predict_score(self,df_test: pd.DataFrame):
        probas = []
        for text in tqdm(df_test['text'].to_list()):
            proba = self.bayesian_pred(text)
            probas.append(proba)
        probas = np.asarray(probas)

        if self.is_ood_label_in_train:
            probas = probas[:,self.clf.trained_classes_mapping.index(self.ood_label)] #get out-domain class probas
        else:
            probas = np.max(probas,axis=1)

        return probas


    def bayesian_pred(self,text):
        seqs = [text.lower()]*15
        seqs = pd.Series(list(map(lambda x: x.lower(), seqs)))
        x_test,_ = build_features(self.feature_extractor,
                                    seqs,seqs,
                                    model=load_feature_extractor(self.feature_extractor))
        probs = self.clf.clf(x_test, training=True)
        mean_probs = tf.reduce_mean(probs, axis=0)
        return mean_probs