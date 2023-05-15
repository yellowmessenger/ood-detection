import json
import pandas as pd
import os
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'relative/path/to/file/you/want')

class DataLoader:
    def __init__(self) -> None:
        pass

    def load(self, name: str) -> pd.DataFrame:
        
        if name == 'clinc150':
            with open(f"{dirname}/raw/clinc150/data_full.json","r") as f_in:
                clinc = json.load(f_in)
            
            df_train = pd.concat([
                pd.DataFrame(clinc['train'],columns=['text','intent']),
                pd.DataFrame(clinc['oos_train'],columns=['text','intent'])
            ])
            df_val = pd.concat([
                pd.DataFrame(clinc['val'],columns=['text','intent']),
                pd.DataFrame(clinc['oos_val'],columns=['text','intent'])
            ])
            df_test = pd.concat([
                pd.DataFrame(clinc['test'],columns=['text','intent']),
                pd.DataFrame(clinc['oos_test'],columns=['text','intent'])
            ])
            return {'train':df_train,'val':df_val,'test':df_test}
        else:
            print(f"data {name} not supported.")
            return