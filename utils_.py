import pandas as pd
import numpy as np
from emoji import UNICODE_EMOJI
from string import punctuation, digits
import re

from tqdm import tqdm
tqdm.pandas()

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
pd.options.plotting.backend = "plotly"

import plotly
import plotly.io as pio
pio.renderers.default = 'jupyterlab'

from nltk import agreement

import random
# import nlpaug.augmenter.char as nac
# import nlpaug.augmenter.word as naw

from google_trans_new import google_translator

def get_training_data(bot_name: str, do_augment: bool = False) -> pd.DataFrame:
    if do_augment:
        try:
            print("Loading augmented data...")
            df_train = pd.read_csv(f"data/train/augmented_{bot_name.lower()}.csv")
        except:
            print("Augmented data not found. Generate new augmented data instead.")
            
            df_train = pd.read_csv(f"data/train/{bot_name.lower()}.csv")
            df_train = df_train[df_train['deleted']==False].reset_index(drop=True) #make sure to get the latest updated utterances 
            df_train = df_train[['text','intent']]
            aug = Augmenter()
            df_train["text"] = df_train['text'].map(aug)
            df_train = df_train.explode("text")
            # perform synonym swapper before preprocessing
            df_train["text"] = df_train["text"].apply(lambda x: replace_synonym(x,bot_name))
            # preprocessing
            df_train["text"] = df_train["text"].apply(lambda x: preprocess_sentence_for_embeddings(x,lowercase_dict[bot_name]))
            # deduplication
            df_train = df_train.drop_duplicates("text")
            df_train.to_csv(f"data/train/augmented_{bot_name.lower()}.csv",index=False)
    else:
        df_train = pd.read_csv(f"data/train/{bot_name.lower()}.csv")
        df_train = df_train[df_train['deleted']==False].reset_index(drop=True) #make sure to get the latest updated utterances 
        df_train = df_train[['text','intent']]
    
    return df_train


def get_prod_data(bot_name: str, translated: bool = False) -> pd.DataFrame:
    df_prod = pd.read_csv(f"data/production/processed_prod_{bot_name}.csv")
    
    #make sure to use the multilingual model, currently this is the only reliable flag
    df_prod = df_prod[df_prod['logs_bot_feature_type']=='xlm_multilingual']
    
    #Drop Masked Message with only '_mask_'
    df_prod = df_prod[df_prod['masked_message'].apply(lambda x: not all(wrd=='_mask_' for wrd in x.split()))]
    
    #Convert masked text to "_"
    df_prod['masked_message'] = df_prod['masked_message'].apply(lambda x: re.sub("_mask_","_",x))
    
    #merge with translated message
    if translated:
        df_translated = pd.read_csv(f"data/production/translated_processed_prod_{bot_name}.tsv",
                                    sep="\t", usecols=["id","translated_message","is_bahasa"])
        df_prod = df_prod.merge(df_translated,left_on=df_prod.index,right_on="id",how="left") \
                         .drop(columns=["id"])
    
    #rename columns
    df_prod = df_prod.rename(columns={'message':'text','masked_message':'masked_text',
                                      'logs_bot_intent':'intent',
                                      'logs_bot_confidence': 'confidence',
                                      'logs_bot_sentiment':'sentiment',
                                     })
    if translated:
        df_prod = df_prod.rename(columns={"translated_message":"translated_text"})
    
    #remove unnecesary columns
    df_prod = df_prod.drop(columns=['bot_id','event','messageType','source','logs_bot_feature_type'])
        
    
    # #confidence score binning
    # df_prod['confidence_level'] = df_prod['confidence'].apply(lambda x: "very high" if x >= 0.95 else
    #                                                                      "high" if x >= 0.85 else
    #                                                                      "moderate" if x >= 0.45 else
    #                                                                      "low" if x >= 0.05 else
    #                                                                      "very low"
    #                                                                   )
    
    # #Add non-alpha column -> text consist only of digit, emoji, and punctuation
    # df_prod['is_nonalpha'] = df_prod['text'].progress_apply(lambda x: nonalpha(x))
    
    #reset index
    df_prod = df_prod.reset_index(drop=True)
    
    return df_prod


def plot_label_distribution(series: pd.Series, show_percentage: bool = False,
                            ready_format: bool = False, order: list = None,
                            figsize: tuple = (12, 8), title: str = None) -> None:
    
    if not ready_format:
        if show_percentage:
            series = (series.value_counts(True) * 100).sort_values(ascending=True)
        else:
            series = (series.value_counts()).sort_values(ascending=True)
            
        if order:
            series = series.reindex(order)
        
    fig = series.plot(kind='barh',
                     height=figsize[0]*100, 
                     width=figsize[1]*100,
                     title=title
                    )
    fig.show()


def nonalpha(string: str) -> bool:
    string = string.strip()
    return is_emoji(string) or string.isdigit() or onlypunctuation(string)


def is_emoji(string: str) -> bool:
    return string in UNICODE_EMOJI['en']


def onlypunctuation(string: str) -> bool:
    return all(char in punctuation for char in string)


def do_eda(bot_name: str, translated: bool, do_augment: bool = False) -> pd.DataFrame:
    print("Get train and prod data\n")
    df_train = get_training_data(bot_name,do_augment)
    df_prod = get_prod_data(bot_name.lower(),translated)
    
    print("Train data info\n")
    display(df_train.info())
    
    print("Prod data info\n")
    display(df_prod.info())
    
    print("="*100)
    print("Distribution of training utterances across intents\n")
    display(df_train['intent'].value_counts().describe())
    num_train_intents = len(df_train['intent'].unique())
    print("Total number of intents: ", num_train_intents)
    if num_train_intents > 50:
        plot_label_distribution(df_train[df_train['intent'].map(df_train['intent'].value_counts()) >= 10]['intent'],
                                title="Training Utterances Distribution (filter out intents with < 10 utterances)",
                                figsize=(12,8)
                               )
    else:
        plot_label_distribution(df_train['intent'],
                                title="Training Utterances Distribution",
                                figsize=(12,8)
                               )
    
    print("="*100)
    print("Distribution of predicted intents in production\n")
    display(df_prod['intent'].value_counts().describe())
    num_prod_intents = len(df_prod['intent'].unique())
    print("Total number of predicted intents in production: ", num_prod_intents)        
    if num_prod_intents > 50:
        plot_label_distribution(df_prod[df_prod['intent'].map(df_prod['intent'].value_counts()) >= 500]['intent'],
                                title="Production Utterances Distribution (filter out intents with < 500 utterances)",
                                figsize=(12,10)
                               )
    else:
        plot_label_distribution(df_prod['intent'],
                                title="Production Utterances Distribution",
                                figsize=(12,10)
                               )
        
    if num_prod_intents < num_train_intents:
        never_predicted_intents = list(set(df_train['intent'].unique()) - set(df_prod['intent'].unique()))
        plot_label_distribution(df_train[df_train['intent'].isin(never_predicted_intents)]['intent'],
                                figsize=(8,10),
                                title="Intents that are never predicted in production"
                               )
    elif num_prod_intents > num_train_intents:
        new_predicted_intents = list(set(df_prod['intent'].unique()) - set(df_train['intent'].unique()))
        plot_label_distribution(df_prod[df_prod['intent'].isin(new_predicted_intents)]['intent'],
                                figsize=(8,10),
                                title="Intents that are not existed during training"
                               )
    
    print("="*100)
    print("Proportion of Bahasa utterances in production\n")
    plot_label_distribution(df_prod['is_bahasa'],
                            show_percentage=True,
                            figsize=(6,8),
                            title="Distribution (%) of Bahasa utterances in production"
                           )
    
    print("="*100)
    print("Proportion of negative sentiments in production\n")
    plot_label_distribution(df_prod['sentiment'],
                            show_percentage=True,
                            order=['negative','neutral','positive'],
                            figsize=(6,8),
                            title="Distribution (%) of sentiments in production"
                           )
    
    print("="*100)
    print("Distribution of confidence level in production\n")
    plot_label_distribution(df_prod['confidence_level'],
                            show_percentage = True,
                            order=['very low','low','moderate','high','very high'],
                            figsize=(6,8),
                            title="Distribution (%) of confidence level in production",
                           )
    
    print("="*100)
    print("Distribution of non-alpha utterances (consists of only digit, emoji, and punctuation)\n")
    plot_label_distribution(df_prod['is_nonalpha'],
                            show_percentage = True,
                            figsize=(6,8),
                            title="Distribution (%) of non-alpha utterances (consists of only digit, emoji, and punctuation)",
                           )
    
    print("="*100)
    print("Relationship between the distribution of predicted intents in production and the distribution of training utterances\n")
    top_n = min(50,num_prod_intents)
    training_utterances_count_per_intent = df_train['intent'].value_counts()

    top_n_prod_intents = pd.Series(df_prod['intent'].value_counts().head(top_n).index)
    top_n_prod_intents = pd.Series(top_n_prod_intents.apply(lambda x: training_utterances_count_per_intent[x] if x in training_utterances_count_per_intent
                                                                      else None
                                                           ).to_list(),
                                   index = top_n_prod_intents)
    top_n_prod_intents.name = "Number of Training Utterances"

    plot_label_distribution(top_n_prod_intents.iloc[::-1],
                            show_percentage = True, ready_format = True,
                            figsize=(12,10),
                            title=f"Top {top_n} ranking of predicted intents in production vs Number of training utterances <br> \
                            (The higher the position vertically the higher the ranking)",
                           )
    
    print("="*100)
    print("Relationship between confidence level and the distribution of training utterances in production\n")
    training_utterances_count_per_intent = df_train['intent'].value_counts()

    df_prod_strip = df_prod.copy()
    df_prod_strip['num_train_utterances'] = df_prod_strip['intent'].apply(lambda x: training_utterances_count_per_intent[x] 
                                                                                    if x in training_utterances_count_per_intent
                                                                                    else None)
    
    #boxplot
    plt.figure(figsize=(8,6))
    sns.boxplot(x="confidence_level", y="num_train_utterances", data=df_prod_strip,
                  order=["very low", "low", "moderate", "high", "very high"]
                 )
    plt.title("Confidence level vs Number of training utterances")
    plt.show()
    
    #jittered strip plot
    plt.figure(figsize=(8,6))
    sns.stripplot(x="confidence_level", y="num_train_utterances", data=df_prod_strip,
                  order=["very low", "low", "moderate", "high", "very high"]
                 )
    plt.title("Confidence level vs Number of training utterances")
    plt.show()
    
    print("="*100)
    print("Relationship between non alpha utterances and confidence score in production\n")
    #boxplot
    plt.figure(figsize=(8,6))
    sns.boxplot(x="is_nonalpha", y="confidence", data=df_prod,
                order=[True,False]
                )
    plt.title("Non-alpha utterances vs Confidence score")
    plt.show()
    
    #jittered strip plot
    plt.figure(figsize=(8,6))
    sns.stripplot(x="is_nonalpha", y="confidence", data=df_prod,
                order=[True,False]
                )
    plt.title("Non-alpha utterances vs Confidence score")
    plt.show()
    
    print("="*100)
    print("Relationship between intent categories and confidence level in production\n")
    top_n = min(50,num_prod_intents)
    top_n_prod_intents = pd.Series(df_prod['intent'].value_counts().head(top_n).index)
    df_prod_top_n_intents = df_prod[df_prod['intent'].isin(top_n_prod_intents)][['confidence','intent']]

    plt.figure(figsize=(12,12))
    sns.stripplot(x="confidence", y="intent", data=df_prod_top_n_intents,
                  order=list(top_n_prod_intents)
                )
    plt.title(f"Top {top_n} ranking of predicted intents in production vs Confidence score \n (The higher the position vertically the higher the ranking)")
    plt.show()
    
    print("="*100)
    print("Relationship between sentiments and confidence level in production\n")
    #boxplot
    plt.figure(figsize=(8,6))
    sns.boxplot(x="sentiment", y="confidence", data=df_prod,
                  order=["negative", "neutral", "positive"]
                 )
    plt.title("Sentiment vs Confidence score")
    plt.show()
    
    #jittered strip plot
    plt.figure(figsize=(8,6))
    sns.stripplot(x="sentiment", y="confidence", data=df_prod,
                  order=["negative", "neutral", "positive"]
                 )
    plt.title("Sentiment vs Confidence score")
    plt.show()
    
    return df_train, df_prod


def calculate_agreements(*raters_label: list) -> None:
    '''https://www.nltk.org/api/nltk.metrics.agreement.html'''
    taskdata = []
    for i, rater_label in enumerate(raters_label):
        taskdata += [[i, idx,label] for idx, label in enumerate(rater_label)]
    ratingtask = agreement.AnnotationTask(data=taskdata)
    print("kappa (averaged naively over pairs) " +str(ratingtask.kappa()))
    print("fleiss (multi_kappa) " + str(ratingtask.multi_kappa()))
    print("alpha (krippendorff's) " +str(ratingtask.alpha()))
    # print("scotts " + str(ratingtask.pi()))


# a function written to consider only ascii characters
def consider_ascii_characters(text):
    text = ''.join([i if ord(i) < 128 else ' ' for i in text])
    return text

def preprocess_sentence_for_embeddings(text,lower_case=False,is_multilingual=True):
    if not is_multilingual:
        text = consider_ascii_characters(text)
    if lower_case:
        text = text.lower()
    if is_multilingual:
        return text
    punctuation_translator = str.maketrans('', '', punctuation)
    digit_translator = str.maketrans('', '', digits)
    return text.translate(punctuation_translator).strip().translate(digit_translator)

def replace_with_regex(text, synonym_mapping, reg):
    match_found = False
    regex = re.compile(reg, re.IGNORECASE)
    # values = re.findall(regex, text.encode('utf-8'), flags=0)
    values = re.findall(regex, text, flags=0)
    if len(values) > 0:
        for value in values:
            if isinstance(value, tuple):
                arr = [a for a in value if a.strip() != ""]
                if len(arr) > 0:
                    match_found = True
                    text = text.replace(arr[0], synonym_mapping[arr[0].lower()])
            else:
                if value != '' and value.lower() in synonym_mapping.keys():
                    match_found = True
                    text = re.sub(r'\b' + re.escape(value) + r'\b', synonym_mapping[value.lower()],
                                  text)

    return text, match_found


def replace_synonym(text: str, bot_name: str) -> str:
    synonyms = synonyms_dict[bot_name]
    all_synonym_words = []
    special_character_words={}
    if synonyms is not None:
        for key in synonyms:
            if key:
                all_synonym_words.append(key)
        if len(all_synonym_words) > 0:
            reg = ''
            # sorting based on the string length to ensure bigger phrases are detected first
            all_synonym_words.sort(reverse=True, key=len)
            for word in all_synonym_words:
                escaped_string = re.escape(word)
                if len(escaped_string) != len(word):
                    special_character_words[" " + word] = " " + synonyms[word]
                    special_character_words[" " + word + " "] = " " + synonyms[word] + " "
                    special_character_words[word + " "] = synonyms[word] + " "
                    continue
                reg = reg + r'\b' + word + r'\b|'
            text, match_found = replace_with_regex(text, synonyms, reg[:-1])
            reg = ''
            for word, value in special_character_words.items():
                reg = reg + r'\b' + re.escape(word) + r'\b|'
            text, match_found = replace_with_regex(text, special_character_words, reg[:-1])
    
    return text


class Augmenter:
    def __init__(self):
        # Intiialize Word Augmenter
        # self.synonym_aug = naw.SynonymAug(aug_src='wordnet', lang='ind')
        self.spelling_aug = naw.SpellingAug()
        self.splitting_aug = naw.SplitAug()
        self.swapping_aug = naw.RandomWordAug(action="swap")
        self.colloq_aug = ColloquialAug()
        # self.back_translation_aug = BackTranslationAug() # via Google Translate API
        # self.back_translation_aug = naw.BackTranslationAug(
        #                                             from_model_name='Helsinki-NLP/opus-mt-id-en', 
        #                                             to_model_name='Helsinki-NLP/opus-mt-en-id'
        #                                             )
        # self.bert_aug = naw.ContextualWordEmbsAug(model_path='indobenchmark/indobert-base-p1', aug_p=0.1)

        # Intialize Character Augmenter
        self.keyboard_aug = nac.KeyboardAug(aug_word_p=0.1,aug_char_max=1)
    
    def __call__(self, text: str) -> list:
        random.seed(0)
        np.random.seed(0)
        
        results = set()
        results.add(text)

        # results.add(self.synonym_aug.augment(text))
        results.add(self.spelling_aug.augment(text))
        results.add(self.splitting_aug.augment(text))
        results.add(self.colloq_aug.augment(text))
        # results.add(self.bert_aug.augment(text))
        results.add(self.keyboard_aug.augment(text))
        
        # back_translated = self.back_translation_aug.augment(text)
        # results.add(back_translated)
        # results.add(self.spelling_aug.augment(back_translated))
        # results.add(self.splitting_aug.augment(back_translated))
        # results.add(self.colloq_aug.augment(back_translated))
        
        for _ in range(3):
            swapped = self.swapping_aug.augment(text)
            results.add(swapped)
            # results.add(self.synonym_aug.augment(swapped))
            results.add(self.spelling_aug.augment(swapped))
            # results.add(self.splitting_aug.augment(swapped))
            results.add(self.colloq_aug.augment(swapped))
            # results.add(self.keyboard_aug.augment(swapped))
    
        return list(results)


class ColloquialAug:
    def __init__(self,aug_p = 0.5, to_informal = True):
        self.aug_p = aug_p
        self.to_informal = to_informal
        
        colloq_mapping = pd.read_csv("/Users/yellow/Desktop/Multilingual Benchmarking/formal_informal.csv",
                                    usecols=["transformed","original-for"])
        self.formal_to_informal_dict = colloq_mapping.set_index("original-for").to_dict()["transformed"]
        self.informal_to_formal_dict = colloq_mapping.set_index("transformed").to_dict()["original-for"]
        
    def augment(self, text):
        splitted_text = text.split()
        replaced_words = []
        to_formal_replace_budget = to_informal_replace_budget = int(self.aug_p * len(splitted_text))
        to_formal_cnt = to_informal_cnt = 0

        if isinstance(self.to_informal,float):
            rule = random.uniform(0,1) < self.to_informal
        elif self.to_informal:
            rule = True
        else:
            rule = False

        for word in splitted_text:
            if rule: 
                if to_informal_cnt < to_informal_replace_budget:
                    if word in self.formal_to_informal_dict:
                        replaced_words.append(self.formal_to_informal_dict[word])

                        to_informal_cnt += 1
                    else:
                        replaced_words.append(word)
                else:
                    replaced_words.append(word)
            else:
                if to_formal_cnt < to_formal_replace_budget:
                    if word in self.informal_to_formal_dict:
                        replaced_words.append(self.informal_to_formal_dict[word])

                        to_formal_cnt += 1
                    else:
                        replaced_words.append(word)
                else:
                    replaced_words.append(word)

        return " ".join(replaced_words)
    
    
class BackTranslationAug:
    '''Back Translator Augmenter via Google Translate API'''
    def __init__(self,lang_src="id",lang_tgt="en"):
        self.lang_src = lang_src
        self.lang_tgt = lang_tgt
        self.translator = google_translator()
        
        
    def augment(self, text):
        back_translated_text = self.translator.translate(self.translator.translate(text,
                                                                         lang_src=self.lang_src,
                                                                         lang_tgt=self.lang_tgt),
                                                    lang_src=self.lang_tgt,
                                                    lang_tgt=self.lang_src
                                                   )

        return back_translated_text
    

import json
with open("/dbfs/mnt/playground_multilingual_benchmarking/synonyms_dict.json","r") as f_in:
    synonyms_dict = json.load(f_in)
    synonyms_dict = {k:v['synonyms'] if 'synonyms' in v else None for k,v in synonyms_dict.items()}
    
with open("/dbfs/mnt/playground_multilingual_benchmarking/synonyms_dict_lion_parcel_customer_balic.json","r") as f_in:
    synonyms_dict_added = json.load(f_in)
    
    for bot_name in ['Lion Parcel (Customer)','BALIC']:
        synonyms_dict[bot_name] = synonyms_dict_added[bot_name]['synonyms'] if 'synonyms' in synonyms_dict_added[bot_name] else None

with open("/dbfs/mnt/playground_multilingual_benchmarking/synonyms_dict_indigo.json","r") as f_in:
    synonyms_dict_added = json.load(f_in)
    synonyms_dict["Indigo"] = synonyms_dict_added["Indigo"]['synonyms'] if 'synonyms' in synonyms_dict_added["Indigo"] else None

with open("/dbfs/mnt/playground_multilingual_benchmarking/synonyms_dict_waste_connections.json","r") as f_in:
    synonyms_dict_added = json.load(f_in)
    synonyms_dict["Waste Connections"] = synonyms_dict_added["Waste Connections"]['synonyms'] if 'synonyms' in synonyms_dict_added["Waste Connections"] else None

with open("/dbfs/mnt/playground_multilingual_benchmarking/trained_intents_dict.json","r") as f_in:
    trained_intents_dict = json.load(f_in)
    lowercase_dict = {k:v['lower_case'] for k,v in trained_intents_dict.items()}
    trained_intents_dict = {k:v['trained_intents'] for k,v in trained_intents_dict.items()}
    
with open("/dbfs/mnt/playground_multilingual_benchmarking/trained_intents_dict_lion_parcel_customer_balic.json","r") as f_in:
    trained_intents_dict_added = json.load(f_in)
    
    for bot_name in ['Lion Parcel (Customer)','BALIC']:
        lowercase_dict[bot_name] = trained_intents_dict_added[bot_name]['lower_case']
        trained_intents_dict[bot_name] = trained_intents_dict_added[bot_name]['trained_intents']

with open("/dbfs/mnt/playground_multilingual_benchmarking/trained_intents_dict_indigo.json","r") as f_in:
    trained_intents_dict_added = json.load(f_in)
    
    lowercase_dict["Indigo"] = trained_intents_dict_added["Indigo"]['lower_case']
    trained_intents_dict["Indigo"] = trained_intents_dict_added["Indigo"]['trained_intents']

with open("/dbfs/mnt/playground_multilingual_benchmarking/trained_intents_dict_waste_connections.json","r") as f_in:
    trained_intents_dict_added = json.load(f_in)
    
    lowercase_dict["Waste Connections"] = trained_intents_dict_added["Waste Connections"]['lower_case']
    trained_intents_dict["Waste Connections"] = trained_intents_dict_added["Waste Connections"]['trained_intents']


bot_ids_dict = {
  "Jd.id": "x1599794252620",
  "Lion Parcel (Customer)": "x1606995828798",
  "BALIC": "x1545488090660",
  "Indigo": "x1563185332983",
  "Waste Connections": "x1594196359981",
  "Oriflame": "x1629455560514",
  "tokko": "x1617718489040",
  "kitabeli": "x1608177436298",
  "AIA": "x1608528672628",
  "tiket.com": "x1628244227861",
}
  