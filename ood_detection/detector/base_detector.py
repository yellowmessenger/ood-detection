import pandas as pd
import numpy as np
from sklearn.metrics import fbeta_score, matthews_corrcoef, precision_score, recall_score
import plotly.express as px

class BaseDetector:
    def __init__(self) -> None:
        pass

    def fit(self):
        pass

    def predict(self,df_test: pd.DataFrame, threshold: float):
        scores = self.predict_score(df_test)

        if self.indomain_is_higher:
            pred = [True if conf > threshold else False for conf in scores]
        else:
            pred = [True if conf < threshold else False for conf in scores]

        return pred

    def predict_score(self):
        pass

    def benchmark(self,df_test: pd.DataFrame, indomain_classes: list,
                  threshold: float):
        if 'intent' not in df_test.columns:
            print("column 'intent' is missing in df_val. Make sure to change your target variable name as 'intent")
            return
        
        if len(indomain_classes)==0:
            print("found empty indomain_classes. Make sure to specify all indomain classes inside a list.")
            return
        
        df_test['is_indomain'] = df_test['intent'].apply(lambda x: x in indomain_classes)
        pred = self.predict(df_test,threshold)

        benchmark_dict = {}
        benchmark_dict['precision'] = precision_score(df_test.is_indomain, pred)
        benchmark_dict['f05'] = fbeta_score(df_test.is_indomain, pred, beta=0.5)
        benchmark_dict['f15'] = fbeta_score(df_test.is_indomain, pred, beta=1.5)
        benchmark_dict['recall'] = recall_score(df_test.is_indomain, pred)
        benchmark_dict['mcc'] = matthews_corrcoef(df_test.is_indomain, pred)

        return benchmark_dict

    def tune_threshold(self, df_val: pd.DataFrame,
                        indomain_classes: list,
                        thresholds = np.linspace(0,5,100)):
        if 'intent' not in df_val.columns:
            print("column 'intent' is missing in df_val. Make sure to change your target variable name as 'intent")
            return
        
        if len(indomain_classes)==0:
            print("found empty indomain_classes. Make sure to specify all indomain classes inside a list.")
            return
        
        df_val['is_indomain'] = df_val['intent'].apply(lambda x: x in indomain_classes)

        # Init visualization data
        df_viz = df_val.copy()
        df_viz['scores'] = self.predict_score(df_val)
        df_viz['scores'] = df_viz['scores'].astype(float)

        # Check metric value for each threshold value
        scores = df_viz['scores'].to_list()
        precision, f05, f15, recall, mcc = [], [], [], [], []
        for score_threshold in thresholds:
            if self.indomain_is_higher:
                pred = [True if conf > score_threshold else False for conf in scores]
            else:
                pred = [True if conf < score_threshold else False for conf in scores]
            
            precision.append(precision_score(df_val.is_indomain, pred))
            f05.append(fbeta_score(df_val.is_indomain, pred, beta=0.5))
            f15.append(fbeta_score(df_val.is_indomain, pred, beta=1.5))
            recall.append(recall_score(df_val.is_indomain, pred))
            mcc.append(matthews_corrcoef(df_val.is_indomain, pred))

        df_viz = pd.DataFrame({
            'Precision': precision,
            'F0.5 Score': f05,
            'MCC': mcc,
            'F1.5 Score': f15,
            'Recall': recall,
        }, index=thresholds)
        df_viz.index.name = "Thresholds"
        df_viz.columns.name = "Rate"

        fig_thresh = px.line(
            df_viz, title='Precision, F0.5, MCC, F1.5, Recall at every threshold',
            width=1000, height=500
        )

        fig_thresh.update_yaxes(range=[-0.5, 1.1], constrain='domain')
        fig_thresh.show()