import os

dataset_path = r"C:\Users\puppy\.cache\kagglehub\datasets\fedesoriano\traffic-prediction-dataset\versions\1"

file_name = "YOUR_FILE.csv"  # replace after checking

df = load_data(os.path.join(dataset_path, file_name))