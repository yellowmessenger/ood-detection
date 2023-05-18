import json
import pandas as pd
import random
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
                ])
            else:
                df_train =  pd.DataFrame(clinc['train'],columns=['text','intent'])

            df_val = pd.concat([
                pd.DataFrame(clinc['val'],columns=['text','intent']),
                pd.DataFrame(clinc['oos_val'],columns=['text','intent'])
            ])
            df_test = pd.concat([
                pd.DataFrame(clinc['test'],columns=['text','intent']),
                pd.DataFrame(clinc['oos_test'],columns=['text','intent'])
            ])

            return {'train':df_train.reset_index(drop=True),
                    'val':df_val.reset_index(drop=True),
                    'test':df_test.reset_index(drop=True)}
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

            return {'train':df_train.reset_index(drop=True),
                    'val':df_val.reset_index(drop=True),
                    'test':df_test.reset_index(drop=True)}
        elif name == 'snips':
            df_train = pd.read_csv(f"{dirname}/raw/snips/snips_train_75_0.csv").rename(columns={'labels':'intent'})
            df_val = pd.read_csv(f"{dirname}/raw/snips/snips_val_75_0.csv").rename(columns={'labels':'intent'})
            df_test = pd.read_csv(f"{dirname}/raw/snips/snips_test_75_0.csv").rename(columns={'labels':'intent'})

            if include_ood_in_train:
                df_train.loc[df_train['is_ood']==1, "intent"] = "oos"
            else:
                df_train = df_train[df_train['is_ood']==0]
            df_val.loc[df_val['is_ood']==1, "intent"] = "oos"
            df_test.loc[df_test['is_ood']==1, "intent"] = "oos"

            return {'train':df_train.reset_index(drop=True),
                    'val':df_val.reset_index(drop=True),
                    'test':df_test.reset_index(drop=True)}
        else:
            print(f"data {name} not supported.")
            return
