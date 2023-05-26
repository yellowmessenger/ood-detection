import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import tensorflow_hub as hub
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertModel
import torch
import gc
from tqdm import tqdm

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
    elif feature_extractor == "bert":
        output = build_bert(model,text_train,intent_train,text_val)
    elif feature_extractor == "bert_best_ckpt":
        output = build_bert(model,text_train,intent_train,text_val)
        output_ckpt = build_bert(model,text_val_ckpt,intent_val_ckpt,None)
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
    elif feature_extractor in ["bert","bert_best_ckpt"]:
        return load_model("bert")
    else:
        print("Feature Extractor's not supported.")
        return
    

def load_model(name: str):
    if name == "use":
        return hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    elif name == "mpnet":
        return SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    elif name == "bert":
        return BertModel.from_pretrained("bert-base-uncased")
    else:
        print("Model's not supported.")
        return
    

def load_tokenizer(name: str):
    if name == "bert":
        return BertTokenizer.from_pretrained('bert-base-uncased')
    else:
        print("Tokenizer's not supported.")
        return


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


def build_bert(model,train_corpus: pd.Series, y_train: pd.Series, val_corpus: pd.Series, batch_size: int = 32) -> np.array:
    #Train corpus embedding
    texts = train_corpus.to_list()
    X_train = []
    for i in tqdm(range(0, len(texts), batch_size)):
        input_ids, attention_mask = generate_indobert_tokens(texts[i:i+batch_size])
        X_train_i = get_bert_embeddings_from_text(model,input_ids, attention_mask, 
                                                        return_whole_output=False)
        X_train.append(X_train_i)
    X_train = np.concatenate(X_train)
    
    if val_corpus is None:
        return X_train, y_train
    else:
        #Val corpus embedding
        input_ids, attention_mask = generate_indobert_tokens(val_corpus.to_list())
        X_val = get_bert_embeddings_from_text(model,input_ids, attention_mask, 
                                                return_whole_output=False)
        return X_train, y_train, X_val


def get_bert_embeddings_from_text(bert_model, input_ids, attention_mask, return_whole_output=False):    
    output_states = bert_model(input_ids,attention_mask)
    output_embeddings = output_states[0]
    pooled_output = pooling(output_embeddings,attention_mask,return_np=True) # mean pooling
    
    del bert_model
    gc.collect()
    
    if return_whole_output is False:
        return pooled_output
    return output_embeddings, attention_mask, pooled_output


def pooling(token_embeddings,attention_mask,return_np=False):
    '''Mean Pooling'''
    output_vectors = []
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = input_mask_expanded.sum(1)
    sum_mask = torch.clamp(sum_mask, min=1e-9)
    output_vectors.append(sum_embeddings / sum_mask)
    output_vector = torch.cat(output_vectors, 1)
    if return_np:
        return np.asarray([emb.cpu().detach().numpy() for emb in output_vector])
    else:
        return output_vector
    

def generate_indobert_tokens(input,max_length=None):
    bert_tokenizer = load_tokenizer("bert")
    
    test_xc = [input] if isinstance(input, str) is True else input
    input_ids, attention_mask = preprocess_for_inference(
        test_xc,
        tokenizer=bert_tokenizer,
        cls_token=bert_tokenizer.cls_token,
        sep_token=bert_tokenizer.sep_token,
        max_length=max_length,
        max_seq_lens_cap=512,
    )
    
    del bert_tokenizer
    gc.collect()
    
    return input_ids, attention_mask


def preprocess_for_inference(
        sentences,
        tokenizer,
        cls_token="[CLS]",
        sep_token="[SEP]",
        pad_token=0,
        max_length=None,
        max_seq_lens_cap=512,
    ):
    all_input_ids = []
    all_attention_mask = []
    if max_length is None:
        max_length=0
        dynamic_padding=True
    else:
        dynamic_padding=False
    sentences = [sentences] if isinstance(sentences, str) is True else sentences

    for sentence in sentences:
        tokens= tokenizer.tokenize(sentence)

        tokens += [sep_token]

        tokens = [cls_token] + tokens
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        if dynamic_padding and len(input_ids) > max_length:
            if len(input_ids) > max_seq_lens_cap:
                max_length = max_seq_lens_cap
            else:
                max_length = len(input_ids)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1] * len(input_ids)
        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)

        # pad the sentences to max length
    for index in range(len(sentences)):
        padding_length = max_length - len(all_input_ids[index])
        all_input_ids[index] += [pad_token] * padding_length
        all_attention_mask[index] += [0] * padding_length
        # padding all original word tokens with empty string
    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    all_attention_mask = torch.tensor(all_attention_mask, dtype=torch.long)

    return all_input_ids, all_attention_mask