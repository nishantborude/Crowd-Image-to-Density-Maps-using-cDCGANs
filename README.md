# Crowd-Image-to-Density-Maps-using-cDCGANs
Currently working with the Computer Vision Lab at Stony Brook University for generating density maps from crowd images

My goal is to build a model that very accurately generates the density maps from crowd images using conditional Deep Convolutional Generative Adversarial Networks.

The project is in the nascent phase.

The current files generate the building facades from the blue stamps. Thus, one can give any sample form of the building facades using different colors representing windows, doors etc and the model generates a realistic building front from it.

The results stored in results folder are generated for only 200 epochs. The approximate running time is 4 hours on GPU.

Sample Test File:


Generated Image:
![cmp_b0205](https://user-images.githubusercontent.com/10834446/37372691-0b69faea-26ea-11e8-9ef0-24e9fa5c5dee.jpg)
