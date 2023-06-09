import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

from ood_detection.detector.trust_scores import TrustScores
from ood_detection.detector.entropy import Entropy
from ood_detection.detector.local_outlier_factor import LOF
from ood_detection.detector.binary_msp import BinaryMSP
from ood_detection.detector.msp import MSP
from ood_detection.detector.doc import DOC
from ood_detection.detector.adb import ADB
from ood_detection.detector.knn import KNN
from ood_detection.detector.mc_dropout import MCDropout
from ood_detection.detector.bnn_vi import BNNVI
from ood_detection.detector.biencoder_cosine import BiEncoderCosine
from ood_detection.detector.biencoder_euclidean import BiEncoderEuclidean
from ood_detection.detector.biencoder_lof import BiEncoderLOF
from ood_detection.detector.biencoder_maha import BiEncoderMaha
from ood_detection.detector.biencoder_entropy import BiEncoderEntropy
from ood_detection.detector.biencoder_pca_entropy import BiEncoderPCAEntropy
from ood_detection.detector.biencoder_pca_cosine import BiEncoderPCACosine
from ood_detection.detector.biencoder_pca_euclidean import BiEncoderPCAEuclidean
from ood_detection.detector.rake import RAKE
from ood_detection.detector.likelihood_ratio import LikelihoodRatio

from ood_detection.dataloaders.base import DataLoader

detector_map = {
                'TrustScores': TrustScores, 
                'Entropy': Entropy, 
                'LOF': LOF, 
                'BinaryMSP':BinaryMSP, 
                'MSP': MSP, 
                'DOC': DOC, 
                'ADB': ADB, 
                'KNN': KNN,
                'MCDropout': MCDropout,
                'BNNVI': BNNVI,
                'BiEncoderCosine': BiEncoderCosine,
                'BiEncoderEuclidean': BiEncoderEuclidean,
                'BiEncoderLOF': BiEncoderLOF,
                'BiEncoderMaha': BiEncoderMaha,
                'BiEncoderEntropy': BiEncoderEntropy,
                'BiEncoderPCAEntropy': BiEncoderPCAEntropy,
                'BiEncoderPCACosine': BiEncoderPCACosine,
                'BiEncoderPCAEuclidean': BiEncoderPCAEuclidean,
                'RAKE': RAKE,
                'LikelihoodRatio': LikelihoodRatio,
            }