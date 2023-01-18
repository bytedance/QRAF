# QRAF: A QUANTIZATION-ERROR-AWARE RATE ADAPTION FRAMEWORK FOR LEARNED IMAGE COMPRESSION
Official implementation of "QRAF: a Quantization-error-aware Rate Adaption Framework for Learned Image Compression"

Environment
Recommend using Miniconda.

# Environment
    #python>=3.6 should be fine.
    conda create -n qraf python=3.8
    conda activate qraf
    pip install compressai==1.1.5
    
# Dataset
```
   mkdir dataset
   mv TestDataset ./dataset
```

# Inference
## Parameters
dataset: dir, Test dataset path.

s: int, discrete bitrate index.

output_path: str, the name of reconstructed dir.

p: str, checkpoint path.

patch: int, padding size.

factormode: int, whether to choose continuous bitrate adaption.

factor: float, continuous bitrate quantization bin size.

## Inference code
```
    python3 Inference.py --dataset ./dataset/TestDataset --s 2 --output_path AttentionVRSTE -p ./Cheng2020VR.pth.tar --patch 64 --factormode 1 --factor 0.1
```

## Note
For discrete bitrate:
```
    python3 Inference.py --dataset ./dataset/TestDataset --s 2 --output_path AttentionVRSTE -p ./Cheng2020VR.pth.tar --patch 64 --factormode 0 --factor 0
```
For continuous bitrate:
Change arbitrary Factor in the range of [0.5, 12] in ./Inference.py line 381.
```
    python3 Inference.py --dataset ./dataset/TestDataset --s 2 --output_path AttentionVRSTE -p ./Cheng2020VR.pth.tar --patch 64 --factormode 1 --factor 0.1
```

## Discrete/Continuous Bitrate Adaption Results
.![](assets/VariableRate.png)

# Variable RD Results
We used the 8 pretrained discrete models for Balle et al., Minnen et al. from [compressai](https://github.com/InterDigitalInc/CompressAI) as the benchmark.

We re-trained 8 Cheng2020 models on our training dataset from low bitrate and high bitrate.
## RD Curve On Kodak dataset with 24 images
### Variable rate of Balle
.![](assets/Balle.png)
### Variable rate of Minnen
.![](assets/Minnen.png)
### Variable rate of Cheng2020
.![](assets/Cheng2020.png)

##Note
From public code and paper, the models of Cheng2020 only trained for the low and medium rate with lambda belonging to {0.0016, 0.0032, 0.0075, 0.015, 0.03, 0.045}. 

We re-trained Cheng2020 on our training dataset following the original paper setting with lambda belonging to {0.0018, 0.0035, 0.0067, 0.0130, 0.0250, 0.0483, 0.0932, 0.1800} for a fair comparison.