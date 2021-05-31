## ESRGAN in Pytorch


* Implementation of Enhanced Super-resolution model in pytorch
* Model runs - [wandb experiment tracker](https://wandb.ai/veb-101/esrgan-pytorch/table?workspace=user-veb-101)
* Some basic details about training process. Apart from the default parameters described in the paper the changes are as follows:
  * Increase adverserial loss factor in stage 2 for 30 epochs then come back to original state.
  * Used Spectral normalization and Two-time Update rule.
  * Two stage training in total 100 and 174 epochs each.
  * LR image of size 64x64 with 4x HR of 256x256.
  * Added FID metric.
  * Training data consists of whole trainig and validation images from Div2k and 1000 images from Flick2k dataset.


* More in-depth details will be updated ASAP. The project has been completed.