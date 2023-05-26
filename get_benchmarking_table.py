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

            # Add metric values
            if os.path.isfile(os.path.join(args.results_dir, filename)):
                with open(os.path.join(args.results_dir, filename),"r") as f_in:
                    res = json.load(f_in)

                for metric in res:
                    temp[metric] = res[metric]
            
            results.append(temp)

    df = pd.DataFrame(results)
    print(df)

    return df
                


if __name__ == "__main__":
    args = parse_arguments()

    df_benchmark = summarize(args)
    df_benchmark.to_csv(f"{args.output_dir}/benchmarking_results.csv",index=False)