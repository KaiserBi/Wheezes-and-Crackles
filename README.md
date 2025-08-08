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

##  Installation
`pip install -r requirements.txt`

##  Data processing 
- The original data files are from https://bhichallenge.med.auth.gr/
- To extract .pt segments with labels, run batch_extraction.py
- Due to the unbalanced amount of wheeze/crackle samples, run balance_crackle.py to balance the amount of both samples

##  Metrics
The system tracks the following:
`F1 score` for labels wheeze and crackle 
`Multi-label accuracy` exact match between prediction and ground truth
`validation loss`

##  Training
-  run train.py for training. Adjust configurations in config.json
-  Using AMP would significantly decrease time taken for every epoch
-  Or you can use my pre-trained model under the 'models' folder
-  the image below shows the train result for 250 epochs for a ResNet with 4 blocks
  ![Train result after 250 epochs, with 4 residual blocks](Result%20plots/Train_result_250_epoch.jpeg)

##  Fine-tuning and Transfer learning
-  for threshold optimization, run optimize_thresholds.py to get the best threshold for labels and the best Multi-lable accuracy for the model
-  use fine_tuning.py for incremental learning. Adjust the configuration through fine_tuning_config.json
-  below shows the result after 100 epochs of fine-tuning, the pre-trained model was trained for 200 epochs
  ![fine-tuning result](Result%20plots/200E+100T_1.jpeg)

-  run transfer_train_resnet6_fixed.py for transferring pre-trained model(4 residual blocks) to 6-residual blocks network
  

