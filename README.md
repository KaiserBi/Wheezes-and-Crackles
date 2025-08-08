# Wheezes-and-Crackles
This project implements a deep learning pipeline for classifying respiratory sounds (crackle and/or wheeze) using a customized 1D ResNet architecture. It includes support for focal loss, threshold optimization, transfer learning, and evaluation based on multi-label accuracy.

## Features
- 1-dimensional ResNet backbone 

- Extra Squeeze-Excitation layer added to the classical ResBlock, improving identification accuracy

- Multi-label classification for simultaneous crackle and wheeze detection

- Focal Loss support for imbalanced audio data

- Threshold optimization to improve multi-label accuracy

- Transfer learning: load 4-block ResNet and transfer to deeper 6-block model

- Metrics tracked: F1 (per label), multi-label accuracy

##  Data_processing 
- The original data files are from https://bhichallenge.med.auth.gr/
- To extract .pt segments with labels, run batch_extraction.py
- Due to the unbalanced amount of wheeze/crackle samples, run balance_crackle.py to balance the amount of both samples

