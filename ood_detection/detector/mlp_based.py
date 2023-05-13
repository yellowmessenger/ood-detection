import pandas as pd
from ood_detection.detector.trust_scores import TrustScore
from ood_detection.classifier.train import train_classifier

class MLP_detector():
    def __init__(self,feature_extractor: str) -> None:
        self.feature_extractor = feature_extractor

    def fit(self,df):
        train_classifier(df, "mlp", self.feature_extractor, 
                        skip_cv = True)

