import pandas as pd
import numpy as np
import math
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import relu
import tensorflow_addons as tfa
import tensorflow_probability as tfp
tfd = tfp.distributions
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import InputExample, losses
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
from ood_detection.classifier.feature_extractor import load_feature_extractor, build_features, load_model

from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


class MLP:
    def __init__(self,feature_extractor: str, use_multi_label: bool = False):
        self.feature_extractor = feature_extractor
        self.use_multi_label = use_multi_label

    def fit(self,x_train: np.array, y: pd.Series,
            x_val: np.array = None, y_val: pd.Series = None,
            **kwargs):
        
        print(f"Fitting MLP with {'multi-label' if self.use_multi_label else 'multi-class'} fashion.")
        
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
                activation="sigmoid" if self.use_multi_label else "softmax",\
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
        
        clf.compile(loss='binary_crossentropy' if self.use_multi_label else 'categorical_crossentropy', 
                    optimizer=adam)
        
        clf.summary()

        history = clf.fit(x_train, y_train,
                      epochs=kwargs.get("epoch",10), 
                      batch_size=32, 
                      validation_split=0, 
                      callbacks=callbacks,
                      validation_data=(x_val, y_val) if ((x_val is not None) and (y_val is not None)) else None, 
                      class_weight=d_class_weights, shuffle=True, verbose=False) 
    
        if (x_val is not None) and (y_val is not None):
            # #plot the training history
            # plt.plot(history.history['loss'], label='Training Loss')
            # plt.plot(history.history['val_loss'], label='Validation Loss')
            # plt.legend()
            # plt.xlabel('Epochs')
            # plt.ylabel('Loss')
            # plt.show()
            
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

class ADBModel:
    def __init__(self,feature_extractor: str):
        self.feature_extractor = feature_extractor

    def fit(self,x_train: np.array, y: pd.Series,
            x_val: np.array = None, y_val: pd.Series = None):
        
        # Prepare Input
        intents_dct = {}
        y_train = y.to_list()
        new_key_value = 0
        for label in y_train:
            if label not in intents_dct.keys():
                intents_dct[label] = new_key_value
                new_key_value += 1

        x_train = tf.convert_to_tensor(x_train, dtype='float32')
        y_train = [intents_dct[label] for label in y_train]
        y_train = tf.convert_to_tensor(y_train, dtype='int32')
        self.intents_dct = intents_dct

        emb_dim = x_train.shape[1]
        clf = ADBPretrainTripletLossModel(emb_dim)
        loss = tfa.losses.TripletSemiHardLoss()
        shuffle = True  # shuffle before every epoch in order to guarantee diversity in pos and neg samples
        batch_size = 300  # same as above - to guarantee...

        clf.compile(optimizer=Adam(learning_rate=1e-4), loss=loss)
        clf.fit(x_train, y_train, epochs=10, shuffle=shuffle, batch_size=batch_size)

        self.clf = clf

    def get_embedding(self,x_test, y_test = None, batch_size: int = 16):
        if isinstance(x_test, pd.Series):
            x_test,y_test = build_features(self.feature_extractor,x_test,y_test,model=load_feature_extractor(self.feature_extractor))
        
        #Prepare input
        x_test = tf.convert_to_tensor(x_test, dtype='float32')
        if y_test is not None:
            y_test = [self.intents_dct[label] for label in y_test]
            y_test = tf.convert_to_tensor(y_test, dtype='int32')

        embeddings_lst = []
        for batch in batches(x_test, batch_size):  # iterate in batches of size 32
            temp_emb = self.clf(batch)
            embeddings_lst.append(temp_emb)

        embeddings = tf.concat(embeddings_lst, axis=0)
        if y_test is not None:
            return embeddings,y_test
        else:
            return embeddings


class ADBPretrainTripletLossModel(tf.keras.Model):
    """Adaptive Decision Boundary with Triplet Loss pre-training model using USE or SBERT embeddings."""

    def __init__(self, emb_dim):
        super(ADBPretrainTripletLossModel, self).__init__()
        self.inp = Input(shape=(emb_dim))
        self.dense = Dense(emb_dim, activation=relu)
        self.dropout = Dropout(0.1)
        self.dense2 = Dense(emb_dim, activation=relu)
        self.dense3 = Dense(emb_dim, activation=relu)
        self.dense4 = Dense(emb_dim, activation=None)

    def call(self, inputs, training=None):
        x = self.dense(inputs)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.dropout(x)
        x = self.dense3(x)

        if training:
            x = self.dropout(x)
            x = self.dense4(x)

        return tf.nn.l2_normalize(x, axis=1)
    
def batches(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]


class MLPDenseFlipout:
    def __init__(self,feature_extractor: str):
        self.feature_extractor = feature_extractor

    def fit(self,x_train: np.array, y: pd.Series,**kwargs):

        x_train,y_train = self.prepare_input(x_train,y)

        bs = 256
        no_cls = len(y.unique())
        kl_divergence_function = (lambda q, p, _: tfd.kl_divergence(q, p) /  # pylint: disable=g-long-lambda
                                                tf.cast(bs, dtype=tf.float32))

        # Init model
        clf = Sequential()
        clf.add(tfp.layers.DenseFlipout(
                                        x_train.shape[1], 
                                        kernel_divergence_fn=kl_divergence_function,
                                        activation=tf.nn.relu, 
                                        input_shape=(x_train.shape[1],)
                                        )
                )
        clf.add(tfp.layers.DenseFlipout(
                                        no_cls, kernel_divergence_fn=kl_divergence_function,
                                        activation=tf.nn.softmax
                                        )
                )

        # Compile and fit model            
        adam = Adam(learning_rate=kwargs.get("adam_learning_rate",1e-3))
        
        clf.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
                    optimizer=adam,
                    experimental_run_tf_function=False,
                    metrics=[tf.keras.metrics.SparseCategoricalCrossentropy(name='crossentropy'),
                                tf.keras.metrics.SparseCategoricalAccuracy()])
        
        clf.summary()

        history = clf.fit(x_train, y_train,
                      epochs=kwargs.get("epoch",10), 
                      batch_size=32, 
                      shuffle=True, verbose=False) 

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
    
    def prepare_input(self,x_train: np.array, y: pd.Series):
        # Get y_train
        self.label_encoder = LabelEncoder()
        y_train = self.label_encoder.fit_transform(y.values)

        return x_train,y_train


class BiEncoder:
    def __init__(self,feature_extractor: str):
        self.feature_extractor = feature_extractor

        if self.feature_extractor not in ['mpnet']:
            raise NotImplementedError("Currently only 'mpnet' is supported. You can add any new sentence-trasnformer model.")

    def fit(self,df_train: pd.DataFrame):
        
        clf = load_model(self.feature_extractor)
        train_examples = list(self.prepare_input(df_train))
        
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
        train_loss = losses.CosineSimilarityLoss(clf)
        clf.fit(train_objectives=[(train_dataloader, train_loss)],
                epochs=1, warmup_steps=10,
                show_progress_bar=True)

        self.clf = clf

    def predict(self,queries: list, df_train: pd.DataFrame):
        train_utterances, train_intents = df_train['text'].to_list(),df_train['intent'].to_list()
        q_embeddings = self.clf.encode(queries)
        train_embeddings = self.clf.encode(train_utterances)
        sims = cosine_similarity(q_embeddings, train_embeddings)
        pred_ids = np.argmax(sims, axis=1)
        return [train_intents[i] for i in pred_ids]
    
    def predict_proba(self,queries: list, train_utterances: list):
        q_embeddings = self.clf.encode(queries)
        train_embeddings = self.clf.encode(train_utterances)
        sims = cosine_similarity(q_embeddings, train_embeddings)
        return np.max(sims, axis=1)
    
    def prepare_input(self,df: pd.DataFrame):
        for i, row in df.iterrows():
            currentSentence = row['text']
            posSentence = df[df.intent==row['intent']].sample(1,random_state=i)['text'].iloc[0]
            negSentence = df[df.intent!=row['intent']].sample(1,random_state=i)['text'].iloc[0]
            yield InputExample(texts=[currentSentence, posSentence], label=1.0)
            yield InputExample(texts=[currentSentence, negSentence], label=0.0)