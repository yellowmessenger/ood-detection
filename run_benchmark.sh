#!/usr/bin bash

for dataset in 'clinc150' 'rostd' 'snips'
do
    for feature_extractor in 'use' 'mpnet' 'bert'
    do
        for use_best_ckpt in False True
        do
            for is_ood_label_in_train in False True
            do
                for detector in 'TrustScores' 'Entropy' 'LOF' 'BinaryMSP' 'MSP' 'DOC' 'ADB' 'KNN' 'MCDropout' 'BNNVI' 'BiEncoderCosine' 'BiEncoderLOF' 'BiEncoderMaha' 'BiEncoderEntropy' 'BiEncoderPCAEntropy' 'BiEncoderPCACosine' 'BiEncoderPCAEuclidean' 'RAKE' 'LikelihoodRatio'
                do
                    output_dir='./benchmarking_results'
                    echo 'Dataset:' $dataset
                    echo 'Feature Extractor:' $feature_extractor
                    echo 'Detector:' $detector
                    echo 'Use Best Checkpoint Model:' $use_best_ckpt
                    echo 'Add OOD in training data:' $is_ood_label_in_train 
                    echo 'Output Dir:' $output_dir
                    python benchmark.py \
                            --output_dir $output_dir \
                            --dataset $dataset \
                            --detector $detector \
                            --feature_extractor $feature_extractor \
                            --use_best_ckpt $use_best_ckpt \
                            --is_ood_label_in_train $is_ood_label_in_train \
                            --ood_label 'oos'
                done
            done
        done
    done
done