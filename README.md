# Uncertainty-Aware Physics-Driven Deep Learning Network for Free-Breathing Liver Fat and R2* Quantification using Self-Gated Stack-of-Radial MRI

UP-Net provides a deep learning framework that rapidly performs artifact suppression and parameter mapping for self-gated 3D stack-of-radial free-breathing liver fat and R2* quantification.

## Overview
Developing, evaluating, and translating DL-based methods for quantitative MRI parameter mapping can be challenging because quantification errors can be difficult to detect by visual inspection. Confidence levels of quantification accuracy from the DL network outputs were not typically characterized. Here, we proposed a Uncertainty-Aware Physics-Driven Deep Learning Network (UP-Net) which suppresses the radial streaking artifacts due to undersampling (motion self-gating), generates quantitative maps (PDFF, R2* and B0 fild maps) and their corresponding uncertainty maps. 

### Input: 
Multi-echo images from undersampled radial data. Real and imaginary components stacked in channel dimension.
### Output: 
(1) Multi-echo images with suppressed radial streaking artifacts. Real and imaginary components stacked in channel dimension. \
(2) Quantitative maps, including PDFF, R2* and B0 field maps. \
(3) Pixel-wsie uncertainty maps for PDFF, R2* and B0 field maps, respectively. \

To cite this paper:
