from pathlib import Path

data_dir = Path("data/chrisi")
output_dir = Path("data/output")
output_dir.mkdir(parents=True, exist_ok=True)
bbox_dir = output_dir/Path("bbox_dir")

model_path = output_dir/Path("model_Random Forest_.pkl")
feature_path = Path("feature/output/gabor_index_1000.csv")
image_dir = data_dir / Path("test_labelled")
deep_learning_out_dir = output_dir / Path(
    "deep_learning_output_MB_ST_SH/test_labelled/")
deep_learning_out_dir.mkdir(parents=True, exist_ok=True)
