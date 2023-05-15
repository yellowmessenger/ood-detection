import pandas as pd
import numpy as np
import tensorflow as tf
import numpy.linalg as LA
import math
from ood_detection.classifier.feature_extractor import load_feature_extractor, build_features
from ood_detection.detector.base import BaseDetector
from sklearn.metrics import fbeta_score,matthews_corrcoef

class ADB(BaseDetector):
    def __init__(self,feature_extractor: str,
                 ood_label: str,
                 alpha: float = 1.0, step_size:float = 0.01) -> None:
        BaseDetector.__init__(self) 
        self.feature_extractor = feature_extractor
        self.alpha = alpha
        self.step_size = step_size
        self.ood_label = ood_label

    def fit(self,df: pd.DataFrame, use_best_ckpt: bool = False):
        # Generate Embeddings
        x_train,y_train= build_features(self.feature_extractor,
                                  df['text'],df['intent'],
                                  model=load_feature_extractor(self.feature_extractor))
        
        x_train = tf.math.l2_normalize(x_train, axis=1)  # normalize to make sure it lies on a unit n-sphere
        self.centroids = compute_centroids(x_train, y_train)
        self.centroids = tf.math.l2_normalize(self.centroids, axis=1)
        self.radius = find_best_radius(x_train, y_train, self.centroids, self.alpha, self.step_size)

    def predict(self,df_test: pd.DataFrame):
        x_test,_ = build_features(self.feature_extractor,
                                  df_test['text'],df_test['text'],
                                  model=load_feature_extractor(self.feature_extractor))
        
        x_test = tf.math.l2_normalize(x_test, axis=1)
        logits = distance_metric(x_test, self.centroids, 'euclidean')
        predictions = tf.math.argmin(logits, axis=1)

        c = tf.gather(self.centroids, predictions)
        d = tf.gather(self.radius, predictions)

        distance = tf.norm(x_test - c, ord='euclidean', axis=1)
        predictions = np.where(distance < d, predictions, self.ood_label)

        return predictions
    
    def predict_score(self):
        raise NotImplementedError("Adaptive Decision Boundary Threshold can only return class prediction.")

    def tune_threshold(self):
        raise NotImplementedError("Adaptive Decision Boundary Threshold can only return class prediction.")
    
    def benchmark(self,df_test: pd.DataFrame):
        if 'intent' not in df_test.columns:
            print("column 'intent' is missing in df_val. Make sure to change your target variable name as 'intent")
            return
        
        df_test['is_outdomain'] = df_test['intent'].apply(lambda x: x==self.ood_label)
        pred = self.predict(df_test)
        pred = [x == self.ood_label for x in pred]

        benchmark_dict = {}
        benchmark_dict['f1'] = fbeta_score(df_test.is_outdomain, pred, beta=1.0)
        benchmark_dict['mcc'] = matthews_corrcoef(df_test.is_outdomain, pred)

        return benchmark_dict


def find_best_radius(X_train, y_train, centroids, alpha, step_size):
    X, y = np.asarray(X_train), np.asarray(y_train)
    centroids = np.asarray(centroids)
    num_classes = len(set(y))  # number of classes
    radius = np.zeros(shape=num_classes)

    for c in range(num_classes):
        dists_sel = LA.norm(X - centroids[c], axis=1)  # distances of every point from the selected centroid

        ood_mask = np.where(y != c, 1, 0)  # out-of-domain
        id_mask = np.where(y == c, 1, 0)  # in-domain
        per = np.sum(ood_mask) / np.sum(id_mask)

        while radius[c] < 2:  # maximum distance on a unit n-sphere is 2
            ood_criterion = (dists_sel - radius[c]) * ood_mask
            id_criterion = (radius[c] - dists_sel) * id_mask

            criterion = tf.reduce_mean(ood_criterion) - (tf.reduce_mean(id_criterion) * per / alpha)

            if criterion < 0:  # ID outweighs OOD
                radius[c] -= step_size
                break

            radius[c] += step_size

    return tf.convert_to_tensor(radius, dtype=tf.float32)


def compute_centroids(X, y):
    X = np.asarray(X)
    y = np.asarray(y)

    emb_dim = X.shape[1]
    classes = set(y)
    num_classes = len(classes)

    centroids = np.zeros(shape=(num_classes, emb_dim))

    for label in range(num_classes):
        embeddings = X[y == label]
        num_embeddings = len(embeddings)

        for emb in embeddings:
            centroids[label, :] += emb

        centroids[label, :] /= num_embeddings

    return tf.convert_to_tensor(centroids, dtype=tf.float32)


def distance_metric(X, centroids, dist_type):
    X = np.asarray(X)
    centroids = np.asarray(centroids)

    num_embeddings = X.shape[0]
    num_centroids = centroids.shape[0]  # equivalent to num_classes

    if dist_type == 'euclidean':
        # modify arrays to shape (num_embeddings, num_centroids, emb_dim) in order to compare them
        x = np.repeat(X[:, np.newaxis, :], repeats=num_centroids, axis=1)
        centroids = np.repeat(centroids[np.newaxis, :, :], repeats=num_embeddings, axis=0)

        logits = tf.norm(x - centroids, ord='euclidean', axis=2)
    else:
        x_norm = tf.nn.l2_normalize(X, axis=1)
        centroids_norm = tf.nn.l2_normalize(centroids, axis=1)
        cos_sim = tf.matmul(x_norm, tf.transpose(centroids_norm))

        if dist_type == 'cosine':
            logits = 1 - cos_sim
        else:  # angular
            logits = tf.math.acos(cos_sim) / math.pi

    return tf.convert_to_tensor(logits)