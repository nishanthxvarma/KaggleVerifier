import os

directories = [
    "tests", 
    "data/real", 
    "data/synthetic", 
    "models", 
    "src/ui", 
    "src/core", 
    "src/ml"
]

base_dir = "C:/Users/nisha/OneDrive/Desktop/kag/KaggleVerifier"

for d in directories:
    os.makedirs(os.path.join(base_dir, d), exist_ok=True)
    
print("Directories created successfully.")
