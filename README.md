# SISSI: Seamless Iterative Semi-Supervised Correction of Imperfect Labels in Microscopy Images

## Introduction
This is the official PyTorch implementation of the paper "[SISSI: Seamless Iterative Semi-Supervised Correction of Imperfect Labels in Microscopy Images](https://marwankefah.github.io/assets/pdf/papers/SISSI.pdf)"
to appear in MICCAI 2022 Workshop on [Domain Adaptation and Representation Transfer DART 2022](https://sites.google.com/view/dart2022/home). 


![SISSI: pipeline](https://github.com/marwankefah/SISSI/blob/master/SISSI_pipeline.png)

## Noisy annotiation generation

The code for noisy annotation generation is in [`noisy_annotations_generation`](https://github.com/marwankefah/cell-segmentation/tree/master/noisy_annotations_generation). Specific algorithms have been developed for different state of cells: dead, alive and inhibited, the noisy image level annotations are assumed to be true when developing these algorithms.  

## Training your model

You can run [`deep_learning_code/train_mix.py`](https://github.com/marwankefah/cell-segmentation/blob/master/deep_learning_code/train_mix.py) to train your model in SSSI framework.

## SISSI Components
### Determining the Start of the Semi-Supervised Phase
- [ADELE](https://github.com/Kangningthu/ADELE) Adoption for Object detection
- The implementation for determining the optimal point
that represents the start of memorisation phase can be found in `if_update` in [`deep_learning_code/utils.py`](cytotoxicity_classification/Classifier.pyhttps://github.com/marwankefah/cell-segmentation/blob/a0ba82a8362ca814c92abd223533d3dbb35e19c2/deep_learning_code/reference/utils.py ).


### Pseudo Label Generation 
- We adopt the implementation of TTA and Weighted Boxes Fusion from [kentaroy47](https://github.com/kentaroy47/ODA-Object-Detection-ttA) for pseudo label generation.
- The code can be found in [`deep_learning_code/odach`](https://github.com/marwankefah/cell-segmentation/tree/a0ba82a8362ca814c92abd223533d3dbb35e19c2/deep_learning_code/odach).
  
### Synthetic-like image adaptation according to pseudo labels (Seamless cloning)

- The use of seamless clone can be found in `seam_less_clone` in [`deep_learning_code/dataloaders/utils`](https://github.com/marwankefah/cell-segmentation/blob/a0ba82a8362ca814c92abd223533d3dbb35e19c2/deep_learning_code/dataloaders/utils.py).


- The iterative update of synthetic-like images is handled by `cell_lab_dataset` in [`deep_learning_code/dataloaders/instance_seg_dataset.py`](https://github.com/marwankefah/cell-segmentation/blob/a0ba82a8362ca814c92abd223533d3dbb35e19c2/deep_learning_code/dataloaders/instance_seg_dataset.py) and the flag to perform this update is controlled in `correct_labels` in [`deep_learning_code/reference/engine.py`](https://github.com/marwankefah/cell-segmentation/blob/a0ba82a8362ca814c92abd223533d3dbb35e19c2/deep_learning_code/reference/engine.py).


<!-- ## Citation
If this code is useful for your research, please consider citing:
```
@article{elbatel2022,
  title={SISSI: Seamless Iterative Semi-Supervised Correction of Imperfect Labels in Microscopy Images},
  author={Marawan Elbatel, Christina Bornberg, Manasi Kattel, Enrique Almar, Claudio Marrocco, Alessandro Bria},
  journal={arXiv preprint arXiv:-----},
  year={2022}
}
``` -->
