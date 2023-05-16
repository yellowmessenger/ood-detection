import json
import pandas as pd
import random
from itertools import chain, combinations
import os
dirname = os.path.dirname(__file__)

random.seed(0)

class DataLoader:
    def __init__(self) -> None:
        pass

    def load(self, name: str, include_ood_in_train: bool) -> pd.DataFrame:
        
        if name == 'clinc150':
            with open(f"{dirname}/raw/clinc150/data_full.json","r") as f_in:
                clinc = json.load(f_in)
            
            if include_ood_in_train:
                df_train = pd.concat([
                    pd.DataFrame(clinc['train'],columns=['text','intent']),
                    pd.DataFrame(clinc['oos_train'],columns=['text','intent'])
                ]).reset_index(drop=True)
            else:
                df_train =  pd.DataFrame(clinc['train'],columns=['text','intent']).reset_index(drop=True)
            df_val = pd.concat([
                pd.DataFrame(clinc['val'],columns=['text','intent']),
                pd.DataFrame(clinc['oos_val'],columns=['text','intent'])
            ]).reset_index(drop=True)
            df_test = pd.concat([
                pd.DataFrame(clinc['test'],columns=['text','intent']),
                pd.DataFrame(clinc['oos_test'],columns=['text','intent'])
            ]).reset_index(drop=True)
            return {'train':df_train,'val':df_val,'test':df_test}
        elif name == 'rostd':
            train_dict = {}
            train_id_temp = pd.read_csv(f"{dirname}/raw/rostd/OODRemovedtrain.tsv",sep='\t',header=None)
            train_id_text = train_id_temp[2].tolist()
            train_id_intent = train_id_temp[0].tolist()

            if include_ood_in_train:
                train_ood_temp = pd.read_csv(f"{dirname}/raw/rostd/OODrelease.tsv",sep='\t',header=None)
                train_ood_text = train_ood_temp[2].tolist()
                train_ood_intent = train_ood_temp[0].map(lambda label: "oos" if label == "outOfDomain" else label).tolist()
                
                train_dict['text'] = train_id_text + train_ood_text
                train_dict['intent'] = train_id_intent + train_ood_intent
            else:
                train_dict['text'] = train_id_text
                train_dict['intent'] = train_id_intent
            df_train = pd.DataFrame(train_dict)


            val_dict = {}
            val_temp = pd.read_csv(f"{dirname}/raw/rostd/eval.tsv",sep='\t',header=None)
            val_dict['text'] = val_temp[2].tolist()
            val_dict['intent'] = val_temp[0].map(lambda label: "oos" if label == "outOfDomain" else label).tolist()
            df_val = pd.DataFrame(val_dict)
            

            test_dict = {}
            test_temp = pd.read_csv(f"{dirname}/raw/rostd/test.tsv",sep='\t',header=None)
            test_dict['text'] = test_temp[2].tolist()
            test_dict['intent'] = test_temp[0].map(lambda label: "oos" if label == "outOfDomain" else label).tolist()
            df_test = pd.DataFrame(test_dict)

            return {'train':df_train,'val':df_val,'test':df_test}
        elif name == 'snips':
            # Reference: https://github.com/huawei-noah/noah-research/blob/master/Maha_OOD/scripts/dataset_preprocess/snips_create_splits.py
            df_train = pd.read_csv(f"{dirname}/raw/snips/train.csv").rename(columns={'label':'intent'})
            df_val = pd.read_csv(f"{dirname}/raw/snips/valid.csv").rename(columns={'label':'intent'})
            df_test = pd.read_csv(f"{dirname}/raw/snips/test.csv").rename(columns={'label':'intent'})

            K = 0.75 #The ID part covers about 75% of the whole dataset. 
            label_space = df_train.intent.value_counts()
            ind_classes = get_snips_splits(label_space.to_dict(), False, K, 5)
            print("\n".join([" ".join(x) for x in ind_classes]))
            
            out = {}
            for num, in_class in enumerate(ind_classes):
                final_train_df, final_val_df, final_test_df = create_final_snips_data(
                    df_train, df_val, df_test, in_class, include_ood_in_train)
                print(len(final_train_df))

                out[num] = {'train':final_train_df,'val':final_val_df,'test':final_test_df}
            return out
        else:
            print(f"data {name} not supported.")
            return


def create_final_snips_data(train_df, val_df, test_df, indomain_classes,
                            include_ood_in_train):
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    if include_ood_in_train:
        train_df.loc[~train_df.intent.isin(indomain_classes), "intent"] = "oos"
    else:
        train_df = train_df[train_df.labels.isin(indomain_classes)]
    val_df.loc[~val_df.intent.isin(indomain_classes), "intent"] = "oos"
    test_df.loc[~test_df.intent.isin(indomain_classes), "intent"] = "oos"

    return train_df, val_df, test_df


def get_snips_splits(label_space, verbose, in_domain_ratio, number_of_splits):
    print("Generating Folders With Unsupervised Splits in the Given Ratio, with ratio being for in-domain domains")
    print("Loading label space")
    if verbose:
        print("Label Space:", label_space)
    label_ids = label_space.keys()
    train_n = sum(label_space.values())
    if verbose:
        print("Generating powersets")
    all_powersets = list(powerset(label_ids))
    if verbose:
        print("Total Number Of Powersets:", len(all_powersets))
    all_powerset_lengths = [sum([label_space[class_name] for class_name in powerset]) for powerset in all_powersets]
    max_domain_ratio = in_domain_ratio * 1.15
    min_domain_ratio = in_domain_ratio * 0.85
    acceptable_powersets = []
    if verbose:
        print("Finding acceptable powersets")
    for i, pset in enumerate(all_powersets):
        if min_domain_ratio * train_n <= all_powerset_lengths[i] <= max_domain_ratio * train_n:
            acceptable_powersets.append(pset)
            if verbose:
                print("Accepted Set:", pset)
                print("Accepted Set Length:", all_powerset_lengths[i], "Total Length:", train_n, "Ratio",
                      all_powerset_lengths[i] / train_n)
                print("Complement Set:", generate_complement(pset, label_ids))
    print("Number Of Accepted Sets:", len(acceptable_powersets))
    random.shuffle(acceptable_powersets)
    acceptable_powersets = acceptable_powersets[:number_of_splits]
    return acceptable_powersets

def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def generate_complement(input_set, universal_set):
    complement_set = []
    for u in universal_set:
        if u not in input_set:
            complement_set.append(u)
    return complement_set