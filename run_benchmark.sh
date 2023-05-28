#!/bin/sh

for dataset in 'snips' 'clinc150' 'rostd'
do
    for detector in 'RAKE' 'LikelihoodRatio'
        do
        output_dir='./benchmarking_results'
        echo 'Dataset:' $dataset
        echo 'Detector:' $detector
        echo 'Output Dir:' $output_dir
        python benchmark.py \
                --output_dir $output_dir \
                --dataset $dataset \
                --detector $detector \
                --ood_label 'oos'
        done
    for feature_extractor in 'use' 'mpnet' 'bert'
    do
        for use_best_ckpt in True False
        do
            for is_ood_label_in_train in False True
            do
                for detector in 'TrustScores' 'Entropy' 'LOF' 'BinaryMSP' 'MSP' 'DOC' 'ADB' 'KNN' 'BiEncoderCosine' 'BiEncoderLOF' 'BiEncoderMaha' 'BiEncoderEntropy' 'BiEncoderPCAEntropy' 'BiEncoderPCACosine' 'BiEncoderPCAEuclidean'
                do
                    if [ \( "$use_best_ckpt" = "True" -a \( "$detector" = "BinaryMSP" -o "$detector" = "DOC" -o "$detector" = "Entropy" -o "$detector" = "LOF" -o "$detector" = "MSP" -o "$detector" = "TrustScores" \) \) -o "$use_best_ckpt" = "False" ]
                    then
                        if [ "$use_best_ckpt" = "True" ]
                        then
                            feature_extractor_temp=$feature_extractor'_best_ckpt'
                        fi
                        output_dir='./benchmarking_results'
                        echo 'Dataset:' $dataset
                        echo 'Feature Extractor:' $feature_extractor_temp
                        echo 'Detector:' $detector
                        echo 'Use Best Checkpoint Model:' $use_best_ckpt
                        echo 'Add OOD in training data:' $is_ood_label_in_train 
                        echo 'Output Dir:' $output_dir
                        python benchmark.py \
                                --output_dir $output_dir \
                                --dataset $dataset \
                                --detector $detector \
                                --feature_extractor $feature_extractor_temp \
                                --use_best_ckpt $use_best_ckpt \
                                --is_ood_label_in_train $is_ood_label_in_train \
                                --ood_label 'oos'
                    fi
                done
            done
        done
    done
done