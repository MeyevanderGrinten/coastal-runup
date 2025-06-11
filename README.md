# coastal-runup

Welcome to Coastal Runup: Python-based methods for runup extraction on dissipative beaches. This repository contains Python code to delineate runup from timestack images with:

1) A color contrast (CC) method based on local entropy (E) and saturation (S), and
2) A machine learning (ML) method based on a simple convolutional neural network (CNN) type architecture informed by five preprocessed input channels of the original timestack image: the grayscale image, I, the
intensity over time, dI/dx, the saturation channel, S, the entropy image, E, and the entropy over time dE/dt.

# Usage

The codes enable users to extract runup values from timestack images under the challenges of highly dissipative conditions. The CC method can be considered applicable for environments showing a sufficient level of turbulence (white foam in the swash) and with an apparent seepage face. The ML method should be considered as a promising proof-of-concept that showed good results when applied at a beach in Galveston, Texas. For application elsewhere, it should be trained to a larger, more diverse set. It holds strong potential for wider applications, especially with further refinement and training on larger datasets, which will improve its robustness and reliability, making it applicable beyond dissipative conditions. If this piques your interest, please read the suggestions for amelioration in van der Grinten et. al (2025).

# Author

Meye Janne van der Grinten 

Contact: meye.vandergrinten@gmail.com

# License

This repository is licensed under the MIT License. See the LICENSE file for more details.

# Citation 

If you use this code in your research, please cite the following paper:
Wave runup extraction on dissipative beaches: new video-based methods (van der Grinten et al., 2025)

BibTex:
{
@article{VANDERGRINTEN2025104757,
title = {Wave runup extraction on dissipative beaches: New video-based methods},
journal = {Coastal Engineering},
volume = {200},
pages = {104757},
year = {2025},
issn = {0378-3839},
doi = {https://doi.org/10.1016/j.coastaleng.2025.104757},
url = {https://www.sciencedirect.com/science/article/pii/S0378383925000626},
author = {Meye J. {van der Grinten} and Jakob C. Christiaanse and Ad J.H.M. Reniers and Falco Taal and Jens Figlus and José A.A. Antolínez},
keywords = {Wave runup, Dissipative, Machine learning, Color contrast, Video imagery},
abstract = {Wave runup observations are important for coastal management providing data to validate predictive models of inundation frequencies and erosion rates, which are vital for assessing the vulnerability of coastal ecosystems and infrastructure. Automated algorithms to extract the instantaneous water line from video imagery struggle under dissipative conditions, where the presence of a seepage face and the lack of contrast between the sand and the swash impede proper extraction, requiring time-intensive data quality control or manual digitization. This study introduces two novel methods, based on color contrast (CC) and machine learning (ML). The CC method combines texture roughness — local entropy — with saturation. Images are first binarized using entropy values and then refined through noise reduction by binarization of the saturation channel. The ML method uses a convolutional neural network (CNN) informed by five channels: the grayscale intensity and its time gradient, the saturation channel, and the entropy and its time gradient. Both methods were validated against nine manually labeled, 80 min video time series. The CC method demonstrated strong agreement with manually digitized water lines (RMSE = 0.12 m, r=0.94 for the vertical runup time series; RMSE = 0.08 m, r=0.97 for the 2% runup exceedance (R2%); and RMSE = 3.88 s, r=0.70 for the mean period (Tm−1,0)). The ML model compared well with the manually labeled time series (RMSE = 0.10 m, r=0.96 for the vertical runup time series; RMSE = 0.09 m, r=0.97 for R2%; and RMSE = 3.51 s, r=0.79 for Tm−1,0). Furthermore, the computed R2% values of both methods show a good agreement with the formula proposed by Stockdon et al. (2006) for extremely dissipative conditions, with RMSE-values lower than 0.13 m and correlations exceeding 0.70 for manual, CC, and ML estimates. While the CC method is deemed applicable for wave-by-wave analysis under similar dissipative conditions with a smooth seepage face and sufficient turbulent swash, the ML method still struggles with new, unseen data. However, it shows promise for a broader application and serves as a viable proof of concept. Together, these methods reduce the need for manual processing and enhance real-time coastal monitoring, contributing to more accurate predictive modeling of runup events and a better understanding of nearshore processes.}
}
}
