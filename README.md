# coastal-runup

Welcome to Coastal Runup: Python-based methods for runup extraction on dissipative beaches. This repository contains Python code to delineate runup from timestack images with:

1) A color contrast (CC) method based on local entropy and saturation, and
2) A machine learning (ML) method based on a simple convolutional neural network (CNN) type architecture informed by five preprocessed input channels of the original timestack image: the grayscale image, I, the
intensity over time, dI/dx, the saturation channel, S, the entropy image, E, and the entropy over time dE/dt.

# Usage

The codes enable users to extract runup values from timestack images under the challenges of highly dissipative conditions. The CC method can be considered applicable for environments showing a sufficient level of turbulence (white foam in swash) and with an apparent seepage face. The ML method should be considered as a promising proof-of-concept that showed good results when applied at a beach in Galveston, Texas. It holds strong potential for wider applications, especially with further refinement and training on larger datasets, which will improve its robustness and reliability, making it applicable beyond dissipative conditions. If this piques your interest, please read the suggestions for amelioration in van der Grinten et. al (2025).

# Author

Meye Janne van der Grinten 

Contact: meye.vandergrinten@gmail.com

# License

This repository is licensed under the MIT License. See the LICENSE file for more details.

# Citation 

If you use this code in your research, please cite the following paper:
Wave runup extraction on dissipative beaches: new video-based methods
Authors
[DOI]
