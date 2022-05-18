# SemiCurv: Semi-Supervised Curvilinear Structure Segmentation

This is the official repository for <a href="">"SemiCurv: Semi-Supervised Curvilinear Structure Segmentation"</a> IEEE TIP 2022.

Recent work on curvilinear structure segmentation has mostly focused on backbone network design and loss engineering. The challenge of collecting labelled data, an expensive and labor intensive process, has been overlooked. While labelled data is expensive to obtain, unlabelled data is often readily available. In this work, we propose SemiCurv, a semi-supervised learning (SSL) framework for curvilinear structure segmentation that is able to utilize such unlabelled data to reduce the labelling burden.

![](Image/framework.png)


This demo code reproduces the results for semi-supervised segmentation on CrackForest, EM and DRIVE datasets.

Please follow the following pipeline to reproduce the results.

## Requirements

The code is tested under Ubuntu 18.04, CUDA 10.2, PyTorch 1.6.0
Install the required packages through

```
pip install -r requirements.txt
```

## Datasets

Prepare datasets for evaluation.

### CrackForest Dataset

Collect CrackForest dataset.

```
mkdir ./Dataset
cd ./Dataset
git clone https://github.com/cuilimeng/CrackForest-dataset.git
```

### EM Dataset

EM128 dataset (cropped patches 128*128 from the original EM dataset [1]).

### DRIVE Dataset

DRIVE128 dataset (cropped patches 128*128 from the original DRIVE dataset [2]).

### Run training script

```
sh train.sh
```

### Adaptation to other datasets
You should notice that data augmentation is very important to the success of MT model.
Adjust the affine transformation parameters in /Trainer/trainer_Unet.py l:89-95 accordingly for your own dataset.

## Reference
[1] I. Arganda-Carreras, S. C. Turaga, D. R. Berger, D. Cireşan, A. Giusti,
L. M. Gambardella, J. Schmidhuber, D. Laptev, S. Dwivedi, J. M.
Buhmann et al., “Crowdsourcing the creation of image segmentation
algorithms for connectomics,” Frontiers in neuroanatomy, 2015.

[2] J. Staal, M. D. Abramoff, M. Niemeijer, M. A. Viergever, and `
B. Van Ginneken, “Ridge-based vessel segmentation in color images
of the retina,” IEEE transactions on medical imaging, 2004.
