# Uncertainty-Aware Physics-Driven Deep Learning Network for Free-Breathing Liver Fat and R2* Quantification using Self-Gated Stack-of-Radial MRI

UP-Net provides a deep learning framework that rapidly performs artifact suppression and parameter mapping for self-gated 3D stack-of-radial free-breathing liver fat and R2* quantification.

## Overview
Developing, evaluating, and translating DL-based methods for quantitative MRI parameter mapping can be challenging because quantification errors can be difficult to detect by visual inspection. Confidence levels of quantification accuracy from the DL network outputs were not typically characterized. Here, we proposed a Uncertainty-Aware Physics-Driven Deep Learning Network (UP-Net) which suppresses the radial streaking artifacts due to undersampling (motion self-gating), generates quantitative maps (PDFF, R2* and B0 fild maps) and their corresponding pixel-wise uncertainty maps. 

## UP-Net architecture
UP-Net contains 3 modules for end-to-end training: \
\
(1) Artifact suppression module: This module suppresses radial streaking artifacts due to undersampling from motion self-gating. \
\
(2) Parameter mapping module: This module takes the multi-echo input image to calculate quantitative PDFF, R2* and B0 field maps. \
\
(3) Uncertainty estimation module: This module calculates uncertainty maps for PDFF, R2* and B0 maps separately. The uncertainty maps can be used to predict errors in the quantification results through careful calibration (see more details in our paper). \
The artifact suppression module is built using UNet. The other two modules are built using a "bifurcated U-Net" because of the need for shared image information between quantitative maps and their uncertainty maps.

### Input: 
Multi-echo images from undersampled radial data. Real and imaginary components stacked in channel dimension.
### Output: 
(1) Multi-echo images with suppressed radial streaking artifacts. Real and imaginary components stacked in channel dimension. \
(2) Quantitative maps, including PDFF, R2* and B0 field maps. \
(3) Pixel-wsie uncertainty maps for PDFF, R2* and B0 field maps, respectively. 

## Usage

UP-Net is an image-based network. The images needs additional pre-processing steps before using UP-Net. Pre-processing steps like self-gating, compressed sensing, and beamforming-based streaking reduction are suggested (see more details in our paper). 

UP-Net was implemented in PyTorch. \
(1) "train_model.py" contains the script for training UP-Net \
(2) Different modules of the network can be found in the folder "models" \
(3) Loss functions including uncertainty loss and physics loss can be found in the folder "loss" 

## To cite this paper
Shih, S-F, Kafali, SG, Calkins, KL, Wu, HH. Uncertainty-aware physics-driven deep learning network for free-breathing liver fat and R2* quantification using self-gated stack-of-radial MRI. Magn Reson Med. 2023; 89: 1567- 1585. doi:10.1002/mrm.29525
