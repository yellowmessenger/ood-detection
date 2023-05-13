import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import tensorflow_hub as hub
from sentence_transformers import SentenceTransformer
import gc

def build_features(feature_extractor: str, 
                   text_train: pd.Series, intent_train: pd.Series,
                   text_val: pd.Series = None, 
                   text_val_ckpt: pd.Series = None, intent_val_ckpt: pd.Series = None,
                   model = None, lowercase: bool = True,
                   **kwargs,
                  ):
    if feature_extractor == "bow":
        output = build_bow(text_train,intent_train,text_val,
                           lowercase=lowercase,**kwargs)
    elif feature_extractor == "tf_idf":
        output = build_tfidf(text_train,intent_train,text_val,
                             lowercase=lowercase,**kwargs)
    elif feature_extractor == "use":
        output = build_use(model,text_train,intent_train,text_val)
    elif feature_extractor == "use_best_ckpt":
        output = build_use(model,text_train,intent_train,text_val)
        output_ckpt = build_use(model,text_val_ckpt,intent_val_ckpt,None)
    elif feature_extractor == "mpnet":
        output = build_mpnet(model,text_train,intent_train,text_val)
    elif feature_extractor == "mpnet_best_ckpt":
        output = build_mpnet(model,text_train,intent_train,text_val)
        output_ckpt = build_mpnet(model,text_val_ckpt,intent_val_ckpt,None)
    else:
        print("Feature Extractor's not supported.")
        return
    
    if "_best_ckpt" not in feature_extractor:
        return output
    else:
        return output, output_ckpt

def load_feature_extractor(feature_extractor:str):
    if feature_extractor in ['bow','tf_idf']:
        return
    elif feature_extractor in ["use","use_best_ckpt"]:
        return load_model("use")
    elif feature_extractor in ["mpnet","mpnet_best_ckpt"]:
        return load_model("mpnet")
    else:
        print("Feature Extractor's not supported.")
        return
    

def load_model(name: str):
    if name == "use":
        return hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    elif name == "mpnet":
        return SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    else:
        print("Model's not supported.")
        return
    

# def load_tokenizer(name: str):
#     if name == "xlm":
#         return XLMRobertaTokenizer.from_pretrained(os.path.join(MODEL_PATH, 'xlm_multilingual_transformers'))
#     else:
#         print("Tokenizer's not supported.")
#         return


def build_bow(train_corpus: pd.Series, y_train: pd.Series, val_corpus: pd.Series,
              lowercase: bool, vectorizer = None,
              **kwargs) -> pd.DataFrame:
    if vectorizer is None:
        vectorizer = CountVectorizer(lowercase=lowercase,**kwargs)
        X_train = vectorizer.fit_transform(train_corpus.to_list()).toarray()
    else:
        X_train = vectorizer.transform(train_corpus.to_list()).toarray()
    
    df_train_bow = pd.DataFrame(X_train,columns=vectorizer.get_feature_names_out())

    if val_corpus is None:
        return df_train_bow, y_train
    else:
        X_val = vectorizer.transform(val_corpus.to_list())
        df_val_bow = pd.DataFrame(X_val.toarray(),columns=vectorizer.get_feature_names_out())
        return df_train_bow, y_train, df_val_bow
    

def build_tfidf(train_corpus: pd.Series, y_train: pd.Series, val_corpus: pd.Series,
                lowercase: bool, vectorizer = None,
                **kwargs) -> pd.DataFrame:
    if vectorizer is None:
        vectorizer = TfidfVectorizer(lowercase=lowercase,**kwargs)
        X_train = vectorizer.fit_transform(train_corpus.to_list()).toarray()
    else:
        X_train = vectorizer.transform(train_corpus.to_list()).toarray()
    
    df_train_tfidf = pd.DataFrame(X_train,columns=vectorizer.get_feature_names_out())

    if val_corpus is None:
        return df_train_tfidf, y_train
    else:
        X_val = vectorizer.transform(val_corpus.to_list())
        df_val_tfidf = pd.DataFrame(X_val.toarray(),columns=vectorizer.get_feature_names_out())
        return df_train_tfidf, y_train, df_val_tfidf


def build_use(model,train_corpus: pd.Series, y_train: pd.Series, val_corpus: pd.Series) -> np.array:
    #Train corpus embedding
    X_train = tf_get_use_embeddings_from_text(model,train_corpus.to_list())
    
    if val_corpus is None:
        return X_train, y_train
    else:
        #Val corpus embedding
        X_val = tf_get_use_embeddings_from_text(model,val_corpus.to_list())
        return X_train, y_train, X_val


def tf_get_use_embeddings_from_text(use_model,input,bs=64):
    # use_model = load_model("use")
    test_xc = [input] if isinstance(input, str) is True else input
    
    output = []
    for i in range(0,len(test_xc),bs):
        output.extend(use_model(test_xc[i:i+bs]).numpy())
    output = np.array(output)

    del use_model
    gc.collect()
    
    return output


def build_mpnet(model,train_corpus: pd.Series, y_train: pd.Series, val_corpus: pd.Series,) -> np.array:
    #Train corpus embedding
    X_train = get_mpnet_embeddings_from_text(model,train_corpus.to_list())
    
    if val_corpus is None:
        return X_train, y_train
    else:
        #Val corpus embedding
        X_val = get_mpnet_embeddings_from_text(model,val_corpus.to_list())
        return X_train, y_train, X_val
    

def get_mpnet_embeddings_from_text(mpnet_model,input):
    
    test_xc = [input] if isinstance(input, str) is True else input
    output = mpnet_model.encode(test_xc)
    
    del mpnet_model
    gc.collect()
    
    return output