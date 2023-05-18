import pandas as pd
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from rake_nltk import Rake
from flashtext import KeywordProcessor
from ood_detection.detector.base import BaseDetector
from sklearn.metrics import fbeta_score,matthews_corrcoef
from gensim.models import FastText

lemmatizer = WordNetLemmatizer()
tokenizer = nltk.RegexpTokenizer(r"\w+")
r = Rake()

class RAKE(BaseDetector):
    def __init__(self,ood_label: str) -> None:
        BaseDetector.__init__(self)
        self.ood_label = ood_label

        print("="*50)
        print("This Detector can only be used when Out-Domain data does not exist in the training data.")
        print("="*50)

    def fit(self,df: pd.DataFrame,enhanced: bool = False):
        if self.ood_label in df['intent'].unique():
            print(f"Found {self.ood_label} in training data. This detector can only be used when Out-Domain data does not exist in the training data.")
            return

        # Generate Keywords        
        train_utterances = preprocess_text(df['text'].tolist())
        self.keywords = get_keywords(train_utterances,enhanced)

    def predict(self,df_test: pd.DataFrame):
        utterances = df_test['text'].to_list()
        processed_text = preprocess_text(utterances)
        predictions = classify_utterances(processed_text, self.keywords)

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

        benchmark_dict = {}
        benchmark_dict['f1'] = fbeta_score(df_test.is_outdomain, pred, beta=1.0)
        benchmark_dict['mcc'] = matthews_corrcoef(df_test.is_outdomain, pred)

        return benchmark_dict


def preprocess_text(utterances: list):
    """
    function for preprocessing of training data and unidentified utterances
    """
    for i in range(len(utterances)):
        utterances[i] = remove_text_with_url(utterances[i]).lower()  # This should be done before spell correct else URL may be broken
        utterances[i] = remove_email(utterances[i])

        utterances[i] = remove_num(utterances[i])

        new_words = tokenizer.tokenize(utterances[i].lower())  # tokenization
        if len(new_words) > 40:  # remove utterances more than length 40 as out of domain
            utterances[i] = ''
            new_words = []

        sentence = ''
        for token, tag in pos_tag(new_words):
            sentence = sentence + lemmatizer.lemmatize(token, reformat_pos(tag[0])) + ' '  # lemmatization with POS tags
        utterances[i] = sentence

    return utterances


def filter_keywords(keyword_list: list):
    """
    function to remove single word keywords that are not present in any multi-word keyword
    """
    multi_keyword = []
    for keyword in keyword_list[:]:
        words = keyword.split()
        if len(words) > 1:
            multi_keyword.extend(words)
    multi_keyword = list(set(multi_keyword))

    for keyword in keyword_list[:]:
        words = keyword.split()
        if len(words) == 1 and keyword not in multi_keyword:
            keyword_list.remove(keyword)

    return keyword_list
  

def filter_stopwords(keyword_list: list):
    """
    function to remove custom stopwords from list of keywords
    """
    stopwords = ['want', 'my', 'send', 'much', 'please', 'plz', 'need', 'know', 'name', 'tell', 'help', 'ok', 'okay', 'is', 'was', 'wa', 'I', 'am', 'were', 'are', 'will', 'should']
    for item in keyword_list[:]:
        if item in stopwords:
            keyword_list.remove(item)
    return keyword_list

  
def get_similar_keywords(training_utterances: list, keyword_list: list):
    model = FastText(vector_size=100, window=3, min_count=1)  # instantiate
    model.build_vocab(corpus_iterable=training_utterances)
    model.train(corpus_iterable=training_utterances, total_examples=len(training_utterances), epochs=5)  # train

    keyword_list = list(map(lambda x: str(x[1]), keyword_list))
    similar_keyword_list = []
    for keyword in keyword_list:
      similar_keyword_list += model.wv.most_similar([keyword],topn=5)
      
    similar_keyword_list = list(map(lambda x: str(x[0]), similar_keyword_list))
    return similar_keyword_list


def get_keywords(training_utterances: list, enhanced: bool = False):
    """
    function to get list of keywords generated by RAKE after some custom filtration
    """
    r.extract_keywords_from_sentences(training_utterances)
    ranked_phrases = r.get_ranked_phrases_with_scores()
    unique_list = list(set(ranked_phrases))
    if enhanced:
      training_utterances = [x.split() for x in training_utterances]
      similar_keywords = get_similar_keywords(training_utterances, unique_list)
      
    unique_list = list(map(lambda x: x[1], unique_list))
    keywords = filter_keywords(unique_list)
    if enhanced:
      keywords = unique_list + similar_keywords
    keywords = filter_stopwords(keywords)
    return keywords
  
def classify_utterances(processed_text: list, keyword_list: list):
    """
    function to identify if the utterances are out-of-domain utterances
    """
    keyword_processor = KeywordProcessor()
    for keyword in keyword_list:
        keyword_processor.add_keyword(keyword)

    def contains_keyword(utterance):
        keywords_found = keyword_processor.extract_keywords(utterance)
        return len(keywords_found) > 0

    output = [not contains_keyword(x) for x in processed_text]

    return output

def reformat_pos(tag):
    """
    get POS tags in the format of wordnet lemmatizer
    """
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # As default pos in lemmatization is Noun
    
def remove_num(inputString: str):
    """remove numbers from string"""
    res = "".join(filter(lambda x: not x.isdigit(), inputString))
    return res
  
def remove_email(inputString: str):
    """remove email ids from string"""
    regex = "\S*[a-zA-Z0-9+_.-]+@[a-zA-Z0-9.-]+\S*"
    res = re.sub(regex, '', str(inputString), flags=re.MULTILINE)
    return res

def remove_text_with_url(inputString: str):
    """return empty string if contains URL"""
    regex = r'\S*https?:\/\/.*[\r\n]*'
    url = re.search(regex, str(inputString), flags=re.MULTILINE)
    if url is None:
        return inputString
    else:
        return ''