import pandas as pd
import numpy as np
from ood_detection.classifier.train import train_classifier
from ood_detection.classifier.feature_extractor import load_feature_extractor, build_features
from ood_detection.detector.base import BaseDetector

class DOC(BaseDetector):
    def __init__(self,feature_extractor: str, is_ood_label_in_train: bool, 
                 ood_label: str = None) -> None:
        BaseDetector.__init__(self) 
        self.feature_extractor = feature_extractor

        # This parameter will be used during inference
        # to know which probability score should be used
        self.is_ood_label_in_train = is_ood_label_in_train
        self.ood_label = ood_label

        if is_ood_label_in_train and ood_label is None:
            print("is_ood_label_in_train is set to True but ood_label is None. Make sure to give the value of ood_label column name.")
            return 

        # This parameter will be used to decide the prediction class
        # If True, the lower the score, the more likely it's outdomain
        # Else, the higher the score, the more likely it's outdomain
        if is_ood_label_in_train:
            self.outdomain_is_lower = False 
        else:
            self.outdomain_is_lower = True     
        
    def fit(self,df: pd.DataFrame, use_best_ckpt: bool = False,
            df_val_ckpt: pd.DataFrame = None):        
        # Fit Classifier
        model_name = "mlp_best_ckpt" if use_best_ckpt else "mlp"
        clf = train_classifier(df, model_name, self.feature_extractor, 
                               df_val_ckpt = df_val_ckpt,
                               skip_cv = True, 
                               use_multi_label=True)

        self.clf = clf

    def predict_score(self,df_test: pd.DataFrame):
        x_test,_ = build_features(self.feature_extractor,
                                  df_test['text'],df_test['text'],
                                  model=load_feature_extractor(self.feature_extractor))
        probas = self.clf.predict_proba(x_test)

        if self.is_ood_label_in_train:
            probas = probas[:,self.clf.trained_classes_mapping.index(self.ood_label)] #get out-domain class probas
        else:
            probas = np.max(probas,axis=1)

        return probas
