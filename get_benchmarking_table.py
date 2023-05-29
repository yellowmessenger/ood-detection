import os
import pandas as pd
import json
import argparse

def parse_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument("--results_dir", default= './benchmarking_results', type=str,
                        help="The directory where all benchmarking results are written.")
    parser.add_argument("--output_dir", default= '.', type=str,
                        help="The directory where the benchmarking table will be saved.")

    args = parser.parse_args()

    return args


def summarize(args):
    results = []
    # Loop through all files in the directory
    for filename in os.listdir(args.results_dir):
        if filename.endswith('.json'):
            temp = {}
            # Parse params
            params = filename.split('.json')[0].split('_')
            if len(params) > 6:
                temp['dataset'],temp['detector'],temp['feature_extractor'],temp['use_best_ckpt'],\
                    temp['is_ood_label_in_train'],temp['ood_label'],temp['adb_alpha'],temp['adb_step_size'] = tuple(params)
            else:
                temp['dataset'],temp['detector'],temp['feature_extractor'],temp['use_best_ckpt'],\
                    temp['is_ood_label_in_train'],temp['ood_label'] = tuple(params)
                
            if len(str(temp['feature_extractor']).split('-'))>1:#_best_ckpt
                temp['feature_extractor'] = str(temp['feature_extractor']).split('-')[0]

            # Add metric values
            if os.path.isfile(os.path.join(args.results_dir, filename)):
                with open(os.path.join(args.results_dir, filename),"r") as f_in:
                    res = json.load(f_in)

                for metric in res:
                    temp[metric] = res[metric]
            
            results.append(temp)

    df = pd.DataFrame(results)
    df = df.sort_values(['dataset','detector','feature_extractor',
                         'use_best_ckpt','is_ood_label_in_train'])
    if all(x in df.columns for x in ['fpr_95','fpr_90']):
        df['fpr_95'] = df.apply(lambda row: row['fpr_95'] if ('fpr95' not in df.columns) or (pd.isnull(row['fpr95'])) else row['fpr95'],axis=1)
        df['fpr_90'] = df.apply(lambda row: row['fpr_90'] if ('fpr90' not in df.columns) or (pd.isnull(row['fpr90'])) else row['fpr90'],axis=1)
        if all(x in df.columns for x in ['fpr95','fpr90']):
            df = df.drop(columns=['fpr95','fpr90'])
    df = df.reset_index(drop=True)
    print(df)

    return df
                


if __name__ == "__main__":
    args = parse_arguments()

    pd.set_option('display.max_rows', None)

    df_benchmark = summarize(args)
    df_benchmark.to_csv(f"{args.output_dir}/benchmarking_results.csv",index=False)
