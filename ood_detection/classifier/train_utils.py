import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight
import optuna
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.backend import clear_session

from ood_detection.classifier.feature_extractor import load_feature_extractor, build_features
from ood_detection.classifier.classifier_head import GaussianNB, MultinomialNB, SVC, RandomForestClassifier, MLP, ADBModel, MLPDenseFlipout, BiEncoder
from ood_detection.classifier.classifier_head import compute_logarithmic_class_weights

def fit_nb(X: pd.DataFrame, y: pd.Series, nb_type: str = "gaussian"):
    if nb_type == "gaussian":
        clf = GaussianNB()
    elif nb_type == "multinomial":
        clf = MultinomialNB()
    else:
        print("only support gaussian or multinomial Naive Bayes")
    
    clf.fit(X,y)
    return clf


def fit_svc(X: pd.DataFrame, y: pd.Series):
    clf = SVC(kernel="rbf",random_state=0,probability=True)
    clf.fit(X,y)
    return clf


def fit_rf(X: pd.DataFrame, y: pd.Series):
    clf = RandomForestClassifier(random_state=0)
    clf.fit(X,y)
    return clf


def fit_mlp(feature_extractor: str, x_train: np.array, y: pd.Series, 
            x_val: np.array = None, y_val: pd.Series = None,
            **kwargs):
    clf = MLP(feature_extractor,use_multi_label=kwargs.get("use_multi_label", False))
    clf.fit(x_train,y,x_val,y_val,**kwargs)
    return clf


def fit_adb(feature_extractor: str, x_train: np.array, y: pd.Series, 
            x_val: np.array = None, y_val: pd.Series = None):
    clf = ADBModel(feature_extractor)
    clf.fit(x_train,y,x_val,y_val)
    return clf


def fit_mlp_dense_flipout(feature_extractor: str, x_train: np.array, y: pd.Series):
    clf = MLPDenseFlipout(feature_extractor)
    clf.fit(x_train,y)
    return clf


def fit_biencoder(feature_extractor: str, df_train: pd.DataFrame):
    clf = BiEncoder(feature_extractor)
    clf.fit(df_train)
    return clf


def hpo_create_model(trial,input_size,output_size):
    model = Sequential()
    model.add(
            Dense(
                input_size, 
                input_shape=(input_size,), 
                activation="relu", 
               )
         )
    n_additional_layers = trial.suggest_int("n_additional_layers", 0, 2)
    for i in range(n_additional_layers):
        num_hidden = trial.suggest_int("n_units_l{}".format(i+2), 100, input_size, log=True)
        do_rate = trial.suggest_float("dropout_rate_l{}".format(i+2),0,0.3)
        model.add(
            Dropout(do_rate)
        )
        model.add(
            Dense(
                num_hidden,
                activation="relu",
            )
        )

    model.add(
        Dense(output_size, 
              activation="sigmoid",
              # kernel_regularizer=l2(weight_decay)
             )
    )
    
    return model


def hpo_create_optimizer(trial):
    # We optimize the choice of optimizers as well as their parameters.
    kwargs = {}
    # optimizer_options = ["RMSprop", "Adam", "SGD"]
    optimizer_options = ["Adam"]
    optimizer_selected = trial.suggest_categorical("optimizer", optimizer_options)
    if optimizer_selected == "RMSprop":
        kwargs["learning_rate"] = trial.suggest_float(
            "rmsprop_learning_rate", 1e-5, 1e-1, log=True
        )
        kwargs["decay"] = trial.suggest_float("rmsprop_decay", 0.85, 0.99)
        kwargs["momentum"] = trial.suggest_float("rmsprop_momentum", 1e-5, 1e-1, log=True)
    elif optimizer_selected == "Adam":
        kwargs["learning_rate"] = trial.suggest_float("adam_learning_rate", 1e-5, 1e-1, log=True)
    elif optimizer_selected == "SGD":
        kwargs["learning_rate"] = trial.suggest_float(
            "sgd_opt_learning_rate", 1e-5, 1e-1, log=True
        )
        kwargs["momentum"] = trial.suggest_float("sgd_opt_momentum", 1e-5, 1e-1, log=True)

    optimizer = getattr(tf.optimizers, optimizer_selected)(**kwargs)
    return optimizer


def hpo_objective(trial, df: pd.DataFrame, feature_extractor: str = "xlm"):
    cv_scores = []
    n_splits = 5
    top_n = 1
    
    feature_model = load_feature_extractor(feature_extractor)
    
    #Duplicate intent with only 1 training utterance
    intent_one = df['intent'].value_counts()
    intent_one = list(intent_one[intent_one==1].index)
    if intent_one:
        print(f"Found {len(intent_one)} intent(s) with only 1 utterance. Will duplicate those intent(s).")
        df = pd.concat([df,df[df['intent'].isin(intent_one)]])
    
    #Perform Stratified K-Fold
    skf = StratifiedKFold(n_splits=n_splits)
    for train_idx, val_idx in skf.split(df,df["intent"]):
        df_train, df_val = df.iloc[train_idx], df.iloc[val_idx]
        
        # Generate Features
        output = build_features(feature_extractor, df_train["text"], df_train["intent"],df_val["text"], 
                               None,None,return_vec=False,model=feature_model)
        if "_mha" not in feature_extractor:
            X_train, y_train, X_val = output
        else:
            X_train, X_train_mask, X_train_pooled, y_train, X_val, X_val_mask, X_val_pooled = output
        
        # Get y_train & y_val
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(y_train)
        onehot_encoder = OneHotEncoder(sparse=False, dtype=np.int32)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded_intents = onehot_encoder.fit_transform(integer_encoded)
        y_train = np.array(onehot_encoded_intents)
        
        integer_encoded = label_encoder.transform(df_val["intent"])
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded_intents = onehot_encoder.transform(integer_encoded)
        y_val = np.array(onehot_encoded_intents)

        # Build model, optimizer, callbacks, and class weights
        model = hpo_create_model(trial,X_train.shape[1],y_train.shape[1]) 
        X_train_input, X_val_input = X_train, X_val
        
        # Callbacks & Others
        optimizer = hpo_create_optimizer(trial)
        callbacks = [ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, min_lr=1e-6, mode='min', verbose=0),
                     optuna.integration.TFKerasPruningCallback(trial, 'val_MatthewsCorrelationCoefficient')
                    ]

        weights_type = trial.suggest_categorical("weights_type", ["lgwt","normal"])
        y_integers = np.argmax(y_train, axis=1)
        if weights_type == "normal":
            class_weights = compute_class_weight('balanced', classes = np.unique(y_integers), y = y_integers)
            d_class_weights = dict(enumerate(class_weights))
        elif weights_type == "lgwt":
            d_class_weights = compute_logarithmic_class_weights(y_integers,
                                                                trial.suggest_float("lgwt_mu", 0.1, 0.6, log=True),
                                                                trial.suggest_float("lgwt_minv", 0.3, 0.7, log=True)
                                                               )

        # Training & Validating
        model.compile(loss='binary_crossentropy', 
                      metrics=[tfa.metrics.MatthewsCorrelationCoefficient(num_classes=y_train.shape[1])],
                      optimizer=optimizer)
        history = model.fit(X_train_input, y_train, 
                              epochs=trial.suggest_int("epoch",15,50), 
                              batch_size=32, 
                              validation_data=(X_val_input, y_val), 
                              callbacks=callbacks,
                              class_weight=d_class_weights, 
                              shuffle=True, 
                              verbose=False)
        
        # Store validation score
        cv_scores.append(history.history["val_MatthewsCorrelationCoefficient"][-1])
        
        # Clear clutter from previous Keras session graphs.
        clear_session()
    
    # Return mean MCC.
    return np.mean(cv_scores)