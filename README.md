# *t*DKI-Net: a joint *q-t* space learning network for diffusion-time-dependent kurtosis imaging

## Introduction

Regular non-learning based fitting of tDKI, such as linear least square fitting, requires densely sampled q-t space (Fig. 1(b1)). Previous learning-based methods primarily focused on investigating the sparsity of the q-space (Fig. 1(b2)), while the tDKI model also has sparsity in t-space and the joint q-t space. In this work, we proposed a joint q-t space downsampling strategy to accelerate tDKI acquisition (Fig. 1(b3)).

<p align="center">
   <img src="./figure/Fitting method.jpg" align="center" width="700">
</p>
<p align="center"> Fig.1 The overall structure of two fitting methods. <p align="center">

This repository provides a demonstration of a q-t space acceleration network for t-DKI model, t-DKI-Net. In this repository we offer an inference framework on three kinds of downsampling mode namely q-1, t-1, and q-t-1 corresponding to our manuscript. The project was originally developed for our work on t-DKI Net and can be used directly or fine-tuned with your dataset. 

<p align="center">
   <img src="./figure/Structure.jpg" align="center" width="700">
</p>
<p align="center"> Fig.2 The overall network structure. <p align="center">


## Requirements

Before you can use this package for image segmentation. You should install the follwing libary at least:
- PyTorch version >=1.8
- Some common python packages such as Numpy, H5py, NiBabel ...

## How to use

1. Compile the requirement library.

2. Download our pretrained models and data from the link: <https://drive.google.com/drive/folders/1Uezgc3m4_CzZUuXN0vmZ5VtA27izV1KS?usp=drive_link>

3. Run our demo, using q-t-1 as an example
    ```
    cd qt0
    
    python test.py
    ```
4. If you want to try q0, just specify input_nc as 210 and output_nc as 9, while for td0, input_nc is 450 and output_nc is 7. 


## Acknowledge and Statement

- This project was designed for academic research, not for clinical or commercial use, as it's a protected patent.
  
- If you have any questions, please feel free to contact [me](mailto:zhengtianshu996@gamil.com).


