from pathlib import Path

data_dir = Path("data/chrisi")
output_dir = Path("data/output")
output_dir.mkdir(exist_ok=True)
bbox_dir = output_dir/Path("bbox_dir")
model_path = output_dir/Path("model.pkl")
