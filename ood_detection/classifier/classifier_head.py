import pandas as pd
import numpy as np
import math
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
from ood_detection.classifier.feature_extractor import load_feature_extractor, build_features

from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


class MLP:
    def __init__(self,feature_extractor: str):
        self.feature_extractor = feature_extractor

    def fit(self,x_train: np.array, y: pd.Series,
            x_val: np.array = None, y_val: pd.Series = None,
            **kwargs):
        
        if x_val is not None and y_val is not None:
            x_train,y_train,x_val,y_val = self.prepare_input(x_train,y,x_val,y_val)
        else:
            x_train,y_train = self.prepare_input(x_train,y)

        # Init model
        clf = Sequential()
        clf.add(
                Dense(
                    x_train.shape[1], 
                    input_shape=(x_train.shape[1],), 
                    activation="relu", 
                )
            )
        n_additional_layers = kwargs.get("n_additional_layers", 0)
        for i in range(n_additional_layers):
            num_hidden = kwargs.get("n_units_l{}".format(i+2), 0)
            do_rate = kwargs.get("dropout_rate_l{}".format(i+2),0)
            clf.add(
                Dropout(do_rate)
            )
            clf.add(
                Dense(
                    num_hidden,
                    activation="relu",
                )
            )
            
        clf.add(
            Dense(y_train.shape[1], 
                activation="softmax",\
                )
        )

        # Compile and fit model
        weights_type = kwargs.get("weights_type","normal")
        y_integers = np.argmax(y_train, axis=1)
        if weights_type == "normal":
            class_weights = compute_class_weight('balanced', classes = np.unique(y_integers), y = y_integers)
            d_class_weights = dict(enumerate(class_weights))
        elif weights_type == "lgwt":
            d_class_weights = compute_logarithmic_class_weights(y_integers,
                                                                kwargs.get("lgwt_mu",0.35),
                                                                kwargs.get("lgwt_minv",0.65))
            
        adam = Adam(learning_rate=kwargs.get("adam_learning_rate",0.01))
        rlr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, min_lr=1e-6, mode='min', verbose=1)
        callbacks = [rlr]
        
        if (x_val is not None) and (y_val is not None):
            filepath = 'tmp.hdf5'
            checkpoint = ModelCheckpoint(filepath=filepath, 
                                        monitor='val_loss',
                                        verbose=0, 
                                        save_best_only=True,
                                        mode='min')
            callbacks.append(checkpoint)
            logs = TensorBoard(log_dir="tmp_logs",
                                histogram_freq=0,
                                write_graph=True,
                                write_images=False,
                                write_steps_per_second=False,
                                update_freq="batch",
                                profile_batch=0,
                                embeddings_freq=0,
                                embeddings_metadata=None,
                            )
            callbacks.append(logs)
        
        clf.compile(loss='categorical_crossentropy', optimizer=adam)
        history = clf.fit(x_train, y_train,
                      epochs=kwargs.get("epoch",25), 
                      batch_size=32, 
                      validation_split=0, 
                      callbacks=callbacks,
                      validation_data=(x_val, y_val) if ((x_val is not None) and (y_val is not None)) else None, 
                      class_weight=d_class_weights, shuffle=True, verbose=False) 
    
        if (x_val is not None) and (y_val is not None):
            #plot the training history
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.show()
            
            print("Loading Best Checkpoint Model...")
            clf = tf.keras.models.load_model(filepath)
            print("Best Checkpoint Model is Loaded!")

        self.clf = clf
        self.trained_classes_mapping = list(self.label_encoder.classes_)
        self.x_train = x_train
        self.y_train = y_train

    def predict(self,x_test, batch_size: int = 16):
        if isinstance(x_test, pd.Series):
            x_test,_ = build_features(self.feature_extractor,x_test,x_test,model=load_feature_extractor(self.feature_extractor))
        probas = self.clf.predict(x_test, batch_size=batch_size)
        pred_ids = np.argmax(probas,axis=1)
        test_pred = [self.trained_classes_mapping[pred_id] for pred_id in pred_ids]
        return test_pred
    
    def predict_ids(self,x_test, batch_size: int = 16):
        if isinstance(x_test, pd.Series):
            x_test,_ = build_features(self.feature_extractor,x_test,x_test,model=load_feature_extractor(self.feature_extractor))
        probas = self.clf.predict(x_test, batch_size=batch_size)
        pred_ids = np.argmax(probas,axis=1)
        return pred_ids
    
    def predict_proba(self,x_test, batch_size: int = 16):
        if isinstance(x_test, pd.Series):
            x_test,_ = build_features(self.feature_extractor,x_test,x_test,model=load_feature_extractor(self.feature_extractor))
        probas = self.clf.predict(x_test, batch_size=batch_size)
        return probas
    
    def prepare_input(self,x_train: np.array, y: pd.Series,
                        x_val: np.array = None, y_val: pd.Series = None):
        # Get y_train
        self.label_encoder = LabelEncoder()
        integer_encoded = self.label_encoder.fit_transform(y)
        onehot_encoder = OneHotEncoder(sparse=False, dtype=np.int32)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded_intents = onehot_encoder.fit_transform(integer_encoded)
        y_train = np.array(onehot_encoded_intents)

        if y_val is not None:
            # Get intents that only existed in the y_train
            feasible_intents = [x for x in y_val.unique() if x in y.unique()]
            
            # Filter x_val and y_val
            y_val = y_val.reset_index(drop=True)
            y_val = y_val[y_val.isin(feasible_intents)]
            x_val = x_val[y_val.index]
            
            # Get y_val
            integer_encoded = self.label_encoder.transform(y_val)
            integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
            onehot_encoded_intents = onehot_encoder.transform(integer_encoded)
            y_val = np.array(onehot_encoded_intents)
            return x_train,y_train,x_val,y_val
        else:
            return x_train,y_train


def compute_logarithmic_class_weights(target, mu=0.35, minv=0.65):
    counter = Counter(target)
    return create_class_weight(counter, mu, minv)


def create_class_weight(labels_dict, mu, minv):
    total = np.sum(list(labels_dict.values()))
    keys = labels_dict.keys()
    class_weight = dict()
    
    for key in keys:
        score = math.log(mu*total/float(labels_dict[key]))
        class_weight[key] = score if score > minv else minv
    
    return class_weight
