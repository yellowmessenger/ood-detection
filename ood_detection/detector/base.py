import pandas as pd
import numpy as np
from sklearn.metrics import fbeta_score, matthews_corrcoef, precision_score, recall_score
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
import plotly.express as px

class BaseDetector:
    def __init__(self) -> None:
        pass

    def fit(self):
        pass

    def predict(self,df_test: pd.DataFrame, threshold: float):
        scores = self.predict_score(df_test)

        if self.outdomain_is_lower:
            pred = [True if conf < threshold else False for conf in scores]
        else:
            pred = [True if conf > threshold else False for conf in scores]

        return pred

    def predict_score(self):
        pass

    def benchmark(self,df_test: pd.DataFrame, indomain_classes: list):
        if 'intent' not in df_test.columns:
            print("column 'intent' is missing in df_val. Make sure to change your target variable name as 'intent")
            return
        
        if len(indomain_classes)==0:
            print("found empty indomain_classes. Make sure to specify all indomain classes inside a list.")
            return
        
        df_test['is_outdomain'] = df_test['intent'].apply(lambda x: x not in indomain_classes)
        pred_scores = self.predict_score(df_test)

        benchmark_dict = {}
        benchmark_dict['fpr_95'] = fpr_n(df_test.is_outdomain, pred_scores, 0.95)
        benchmark_dict['fpr_90'] = fpr_n(df_test.is_outdomain, pred_scores, 0.90)
        benchmark_dict['aupr'] = aupr(df_test.is_outdomain, pred_scores)
        benchmark_dict['auroc'] = auroc(df_test.is_outdomain, pred_scores)

        return benchmark_dict

    def tune_threshold(self, df_val: pd.DataFrame, indomain_classes: list):
        if 'intent' not in df_val.columns:
            print("column 'intent' is missing in df_val. Make sure to change your target variable name as 'intent")
            return
        
        if len(indomain_classes)==0:
            print("found empty indomain_classes. Make sure to specify all indomain classes inside a list.")
            return
        
        df_val['is_outdomain'] = df_val['intent'].apply(lambda x: x not in indomain_classes)

        # Init visualization data
        df_viz = df_val.copy()
        df_viz['scores'] = self.predict_score(df_val)
        df_viz['scores'] = df_viz['scores'].astype(float)

        # Check metric value for each threshold value
        scores = df_viz['scores'].to_list()
        thresholds = np.linspace(df_viz['scores'].min() - 0.1,df_viz['scores'].max() + 0.1,100)
        precision, f05, f15, recall, mcc = [], [], [], [], []
        for score_threshold in thresholds:
            if self.outdomain_is_lower:
                pred = [True if conf < score_threshold else False for conf in scores]
            else:
                pred = [True if conf > score_threshold else False for conf in scores]
            
            precision.append(precision_score(df_val.is_outdomain, pred))
            f05.append(fbeta_score(df_val.is_outdomain, pred, beta=0.5))
            f15.append(fbeta_score(df_val.is_outdomain, pred, beta=1.5))
            recall.append(recall_score(df_val.is_outdomain, pred))
            mcc.append(matthews_corrcoef(df_val.is_outdomain, pred))

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


def fpr_n(y_true: np.array, y_scores: np.array, n: float = 0.95):
    fpr,tpr,_ = roc_curve(y_true,y_scores)

    fpr0=0
    tpr0=0
    for i,(fpr1,tpr1) in enumerate(zip(fpr,tpr)):
        if tpr1>=n:
            break
        fpr0=fpr1
        tpr0=tpr1
    fpr_n = ((n-tpr0)*fpr1 + (tpr1-n)*fpr0) / (tpr1-tpr0)
    return fpr_n


def aupr(y_true: np.array, y_scores: np.array):
    return average_precision_score(y_true, y_scores)


def auroc(y_true: np.array, y_scores: np.array):
    return roc_auc_score(y_true, y_scores)
