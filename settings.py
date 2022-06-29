from pathlib import Path

cell_lab_data_dir = Path("../data/cell_lab")

# noisy annotations output folder from Image processing algorithms
noisy_bbox_output_dir = Path("annotations")

###for cell cyto classification
noisy_annotations_generation_path = Path('../noisy_annotations_generation/')
model_path = Path("cytotoxicity_classification/ML_Results/model_Random Forest_.pkl")
feature_path = Path("cytotoxicity_classification/gabor_selected_indices/gabor_index_1000.csv")
image_dir = Path("data/cell_lab") / Path("test_labelled")
deep_learning_out_dir = Path(
    "noisy_annotations_generation/deep_learning_output/test_labelled/")

