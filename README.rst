======================================
Tensorflow "Bayesian U-Net" aka BUNet
======================================

This is the source code for the MICCAI 2018 Paper, **Exploring Uncertainty Measures in Deep Networks for Multiple Sclerosis Lesion Detection and Segmentation** (`Nair et al. <https://arxiv.org/abs/1808.01200>`_), of which I am the first author. 

The network architecture is a heavily modified **U-Net** (`Ronneberger et al. <https://arxiv.org/pdf/1505.04597.pdf>`_), developed in Tensorflow. The network is augmented to provide the following 4 different uncertainty measures as an output.  

1. Mutual Information (`Gal et al. <https://arxiv.org/abs/1703.02910>`_)

2. Entropy (`Gal et al. <https://arxiv.org/abs/1703.02910>`_)

3. MC Sample Variance (`Leibig et al. <https://www.ncbi.nlm.nih.gov/pubmed/29259224>`_)

4. Predicted Variance (`Kendall and Gal <https://arxiv.org/abs/1703.04977>`_)

Details about the network architecture, and the equations for the uncertainty measures can be found in the paper here: https://arxiv.org/abs/1808.01200

The dataset used for this project comes from a large, proprietary, multi-site, multi-scanner, clinical MS dataset. As such, to use this code you will have to modify the dataprovider to be specific to your dataset. 

**Training**:

1. pip install -r requirements.txt

2. python bunet_launcher.py -o ./path_to_output/ -c bunet/configs/train_bunet.json


Author: Tanya Nair
