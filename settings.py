from pathlib import Path

data_dir = Path("../data/chrisi")
output_dir = Path("../feature/data/output")
output_dir.mkdir(exist_ok=True)
bbox_dir = output_dir/Path("bbox_dir")

model_path = output_dir/Path("model_rf_intersect.pkl")
feature_path = Path("../feature/intersect_gabor_1.csv")
image_dir = data_dir / Path("test_labelled")
deep_learning_out_dir = data_dir / Path(
    "weak_labels_reduced_nms/deep_learning_output/deep_learning_output_MB_ST_SH/test_labelled/")
