# cell-segmentation

## Running feature extraction pipeline

### Cropping and glcm features
1. Make sure that the paths `image_dir` and `bbox_dir` in `feature/extract_features.py` are correct. Path `bbox_dir` should contain .txt files with bounding boxes for all cell types in the same dir.
2. Run `python -m feature.extract_features` in your terminal
3. Cropped images will be saved to `data/cropped` and the glcm features will be saved to `data/output`

### Gabor features
1. Run `python -m feature.gabor_filters` in your terminal.
2. Index of top 200 best features selected by AdaBoost will be saved to `feature/output/gabor_index.csv`
