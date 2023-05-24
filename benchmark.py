import os
import json
import argparse
from ood_detection import detector_map, DataLoader

def parse_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", default= './benchmarking_results', type=str, 
                        required = True,
                        help="The output directory where all benchmarking results will be written.")

    parser.add_argument("--dataset", choices=['clinc150','rostd','snips'], type=str, 
                        required = True,
                        help="The name of the dataset to train selected")

    parser.add_argument("--detector", choices=['TrustScores','Entropy','LOF',
                                             'BinaryMSP','MSP','DOC','ADB',
                                             'KNN','MCDropout','BNNVI',
                                             'BiEncoderCosine','BiEncoderLOF',
                                             'BiEncoderMaha','BiEncoderEntropy',
                                             'BiEncoderPCAEntropy','BiEncoderPCACosine',
                                             'BiEncoderPCAEuclidean','RAKE',
                                             'LikelihoodRatio'
                                             ], 
                        type=str,
                        required = True,
                        help="which detector to use")
    
    parser.add_argument("--feature_extractor", choices=['mpnet','use','bert'], 
                        type=str,
                        required = True,
                        help="which feature extractor to use")
    
    parser.add_argument("--use_best_ckpt", type=bool, default=False,
                        help="whether to use best checkpoint of the classifier based on validation data")
    
    parser.add_argument("--is_ood_label_in_train", type=bool, default=True,
                        help="whether to add ood label in the training data")
    
    parser.add_argument("--ood_label", default = 'oos', type=str, required = True,
                        help="name of the ood label")

    parser.add_argument("--adb_alpha", type=float, default=0.75,
                        help="alpha hyperparameter for ADB detector")
    
    parser.add_argument("--adb_step_size", type=float, default=0.01,
                        help="step_size hyperparameter for ADB detector") 

    args = parser.parse_args()

    return args


def run_benchmark(args):
    #Loading Data
    data = DataLoader().load(args.dataset,args.is_ood_label_in_train)

    #Initiate Detector
    if args.detector in detector_map:
        detector = detector_map[args.detector]
    else:
        print(f"{args.detector} is not supported. Supported detectors are: {detector_map.keys()}")
        return
    
    if args.detector in ['BiEncoderCosine','BiEncoderLOF','BiEncoderMaha',
                         'BiEncoderEntropy','BiEncoderPCAEntropy',
                         'BiEncoderPCACosine','BiEncoderPCAEuclidean',
                         'BinaryMSP','Entropy','KNN','LOF','TrustScores']:
        detector = detector(args.feature_extractor)
    elif args.detector == 'ADB':
        detector = detector(args.feature_extractor,args.ood_label,
                            args.adb_alpha,args.adb_step_size
                            )
    elif args.detector in ['BNNVI','DOC','MCDropout','MSP']:
        detector = detector(args.feature_extractor,args.is_ood_label_in_train,
                            args.ood_label)
    elif args.detector == 'LikelihoodRatio':
        detector = detector(args.dataset)
    elif args.detector == 'RAKE':
        detector = detector(args.ood_label)

    #Fitting the detector
    if args.detector in ['BiEncoderCosine','BiEncoderLOF','BiEncoderMaha',
                         'BiEncoderEntropy','BiEncoderPCAEntropy',
                         'BiEncoderPCACosine','BiEncoderPCAEuclidean',
                         'ADB','BNNVI','KNN','RAKE']:
        detector.fit(data['train'])
    elif args.detector == 'BinaryMSP':
        if args.use_best_ckpt:
            detector.fit(data['train'],args.ood_label,True,data['val'])
        else:
            detector.fit(data['train'],args.ood_label)
    elif args.detector in ['DOC','Entropy','LOF','MCDropout','MSP','TrustScores']:
        if args.use_best_ckpt:
            detector.fit(data['train'],True,data['val'])
        else:
            detector.fit(data['train'])
    elif args.detector == 'LikelihoodRatio':
        detector.fit()

    # Benchmark on test data
    if args.detector in ['ADB','RAKE']:
        return detector.benchmark(data['test'])
    elif args.detector == 'LikelihoodRatio':
        return detector.benchmark()
    else:
        return detector.benchmark(data['test'],args.ood_label)
        

if __name__ == '__main__':
    args = parse_arguments()

    # Preparing Output Path
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"{args.output_dir} directory is created!")

    output_path = f"{args.output_dir}/{args.dataset}_{args.detector}_" +\
                    f"{args.feature_extractor}_{args.use_best_ckpt}_{args.is_ood_label_in_train}_" +\
                    f"{args.ood_label}"
    if args.detector == 'ADB':
        output_path += f'_{args.adb_alpha}_{args.adb_step_size}'

    # Running the benchmark
    benchmark_dict = run_benchmark(args)

    # Save the benchmarking results
    with open(output_path+".json","w") as f_out:
        json.dump(benchmark_dict,f_out)