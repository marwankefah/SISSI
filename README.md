# cell-segmentation
## Image processing algorithms

### Alive cells detection
The method for alive cells detection is in [`alive_cells/alive_cells_bboxes.py`](https://github.com/marwankefah/cell-segmentation/blob/master/alive_cells/alive_cells_bboxes.py).

### Dead cells detection
The method for dead cells detection is in [`dead_cells/dead_cells_bboxes.py`](https://github.com/marwankefah/cell-segmentation/blob/master/dead_cells/dead_cells_bboxes.py).

### Inhibited cells detection
The method for inhibited cells detection is in [`inhib_cells/inhib_cells_bboxes.py`](https://github.com/marwankefah/cell-segmentation/blob/master/inhib_cells/inhib_cells_bboxes.py).

## Running feature extraction pipeline
### Cropping and glcm features
1. Make sure that the paths `image_dir` and `bbox_dir` in [`feature/extract_features.py`](https://github.com/marwankefah/cell-segmentation/blob/master/feature/extract_features.py) are correct. Path `bbox_dir` should contain .txt files with bounding boxes for all cell types in the same dir.
2. Run `python -m feature.extract_features` in your terminal.
3. Cropped images will be saved to `data/cropped` and the glcm features will be saved to `data/output`.

### Gabor features
1. Run `python -m feature.gabor_filters` in your terminal.
2. Index of top 1000 best features selected by AdaBoost will be saved to `feature/output/gabor_index.csv`.

## Training the Machine Learning Cell State classifier
The training pipeline for cell state classification is in [`feature/training.py`](https://github.com/marwankefah/cell-segmentation/blob/master/feature/extract_features.py). The model will be saved to `model_path` specified in [`settings.py`](https://github.com/marwankefah/cell-segmentation/blob/master/settings.py).

## Inference Pipeline for Cell state classifier
To run the inference pipeline, add path to the chosen bounding boxes from Deep Learning in [`settings.py`](https://github.com/marwankefah/cell-segmentation/blob/master/settings.py), and run `python -m src.run_nms_before` in your terminal. The inference pipeline is in [`src/run_nms_before`](https://github.com/marwankefah/cell-segmentation/blob/master/src/run_nms_before.py).

