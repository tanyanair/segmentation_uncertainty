======================================
Tensorflow "Bayesian U-Net" aka BUNet
======================================

This is a heavily modified **U-Net** implementation (`Ronneberger et al. <https://arxiv.org/pdf/1505.04597.pdf>`_) developed in Tensorflow.
The network is augmented to provide 4 different uncertainty measures as an output.

1. Mutual Information (`Gal et al. <https://arxiv.org/abs/1703.02910>`_)

2. Entropy (`Gal et al. <https://arxiv.org/abs/1703.02910>`_)

3. MC Sample Variance (`Leibig et al. <https://www.ncbi.nlm.nih.gov/pubmed/29259224>`_)

4. Predicted Variance (`Kendall and Gal <https://arxiv.org/abs/1703.04977>`_)

**Training**:

1. pip install -r requirements.txt

2. python bunet_launcher.py -o ./path_to_output/ -c bunet/configs/train_bunet.json

Author: Tanya Nair
