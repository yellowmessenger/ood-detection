import pandas as pd
import numpy as np
from ood_detection.classifier.train import train_classifier
from ood_detection.classifier.feature_extractor import load_feature_extractor, build_features
from sklearn.metrics import fbeta_score, matthews_corrcoef, precision_score, recall_score
import plotly.express as px
import plotly.io as pio

class TrustScores():
    def __init__(self,feature_extractor: str) -> None:
        self.feature_extractor = feature_extractor

    def fit(self,df: pd.DataFrame, use_best_ckpt: bool = False):
        # Fit Classifier
        model_name = "mlp" if not use_best_ckpt else "mlp_best_ckpt"
        clf = train_classifier(df, model_name, self.feature_extractor, skip_cv = True)

        # Initialize trust score.
        trust_model = TrustScore()
        trust_model.fit(clf.x_train, np.array([clf.trained_classes_mapping.index(x) for x in df['intent'].to_list()]))

        self.trust_model = trust_model
        self.clf = clf

    def predict(self,df_test: pd.DataFrame, return_cls_pred: bool = False):
        x_test,_ = build_features(self.feature_extractor,
                                  df_test['text'],df_test['text'],
                                  model=load_feature_extractor(self.feature_extractor))
        pred_ids = self.clf.predict_ids(x_test)
        if return_cls_pred:
            test_pred = [self.clf.trained_classes_mapping[pred_id] for pred_id in pred_ids]

        # Compute trusts score, given (unlabeled) testing examples and (hard) model predictions.
        trust_score = self.trust_model.get_score(x_test, pred_ids)

        if return_cls_pred:
            return trust_score,test_pred
        else:
            return trust_score
    
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

        #Get Trust Score
        trust_score = self.predict(df_val)

        # Init visualization data
        df_viz = df_val.copy()
        df_viz['trust_score'] = trust_score
        df_viz['trust_score'] = df_viz['trust_score'].astype(float)

        # Check metric value for each threshold value
        confs = df_viz['trust_score'].to_list()
        precision, f05, f15, recall, mcc = [], [], [], [], []
        for probas_threshold in thresholds:
            pred = [True if conf > probas_threshold else 0 for conf in confs]
            
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



#Reference: https://github.com/google/TrustScore/blob/master/trustscore/trustscore.py

# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from sklearn.neighbors import KDTree, KNeighborsClassifier


class TrustScore:
    """
    Trust Score: a measure of classifier uncertainty based on nearest neighbors.
  """

    def __init__(self, k=10, alpha=0.0, filtering="none", min_dist=1e-12):
        """
        k and alpha are the tuning parameters for the filtering,
        filtering: method of filtering. option are "none", "density",
        "uncertainty"
        min_dist: some small number to mitigate possible division by 0.
    """
        self.k = k
        self.filtering = filtering
        self.alpha = alpha
        self.min_dist = min_dist

    def filter_by_density(self, X: np.array):
        """Filter out points with low kNN density.
    Args:
    X: an array of sample points.
    Returns:
    A subset of the array without points in the bottom alpha-fraction of
    original points of kNN density.
    """
        kdtree = KDTree(X)
        knn_radii = kdtree.query(X, k=self.k)[0][:, -1]
        eps = np.percentile(knn_radii, (1 - self.alpha) * 100)
        return X[np.where(knn_radii <= eps)[0], :]

    def filter_by_uncertainty(self, X: np.array, y: np.array):
        """Filter out points with high label disagreement amongst its kNN neighbors.
    Args:
    X: an array of sample points.
    Returns:
    A subset of the array without points in the bottom alpha-fraction of
    samples with highest disagreement amongst its k nearest neighbors.
    """
        neigh = KNeighborsClassifier(n_neighbors=self.k)
        neigh.fit(X, y)
        confidence = neigh.predict_proba(X)
        cutoff = np.percentile(confidence, self.alpha * 100)
        unfiltered_idxs = np.where(confidence >= cutoff)[0]
        return X[unfiltered_idxs, :], y[unfiltered_idxs]

    def fit(self, X: np.array, y: np.array):
        """Initialize trust score precomputations with training data.
    WARNING: assumes that the labels are 0-indexed (i.e.
    0, 1,..., n_labels-1).
    Args:
    X: an array of sample points.
    y: corresponding labels.
    """

        self.n_labels = np.max(y) + 1
        self.kdtrees = [None] * self.n_labels
        if self.filtering == "uncertainty":
            X_filtered, y_filtered = self.filter_by_uncertainty(X, y)
        for label in range(self.n_labels):
            if self.filtering == "none":
                X_to_use = X[np.where(y == label)[0]]
                self.kdtrees[label] = KDTree(X_to_use)
            elif self.filtering == "density":
                X_to_use = self.filter_by_density(X[np.where(y == label)[0]])
                self.kdtrees[label] = KDTree(X_to_use)
            elif self.filtering == "uncertainty":
                X_to_use = X_filtered[np.where(y_filtered == label)[0]]
                self.kdtrees[label] = KDTree(X_to_use)

            if len(X_to_use) == 0:
                print(
                    "Filtered too much or missing examples from a label! Please lower "
                    "alpha or check data."
                )

    def get_score(self, X: np.array, y_pred: np.array):
        """Compute the trust scores.
    Given a set of points, determines the distance to each class.
    Args:
    X: an array of sample points.
    y_pred: The predicted labels for these points.
    Returns:
    The trust score, which is ratio of distance to closest class that was not
    the predicted class to the distance to the predicted class.
    """
        d = np.tile(None, (X.shape[0], self.n_labels))
        for label_idx in range(self.n_labels):
            try:
              d[:, label_idx] = self.kdtrees[label_idx].query(X, k=2)[0][:, -1]
            except:
              d[:, label_idx] = self.kdtrees[label_idx].query(X, k=1)[0][:, -1]

        sorted_d = np.sort(d, axis=1)
        d_to_pred = d[range(d.shape[0]), y_pred]
        d_to_closest_not_pred = np.where(
            sorted_d[:, 0] != d_to_pred, sorted_d[:, 0], sorted_d[:, 1]
        )
        return d_to_closest_not_pred / (d_to_pred + self.min_dist)
