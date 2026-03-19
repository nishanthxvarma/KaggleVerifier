import pandas as pd
import numpy as np

def generate_heart_disease_proxy():
    # Synthetic but realistic-looking UCI Heart Disease proxy
    np.random.seed(42)
    n = 303
    data = {
        'age': np.random.randint(29, 78, n),
        'sex': np.random.randint(0, 2, n),
        'cp': np.random.randint(0, 4, n),
        'trestbps': np.random.randint(94, 201, n),
        'chol': np.random.randint(126, 565, n),
        'fbs': np.random.randint(0, 2, n),
        'restecg': np.random.randint(0, 3, n),
        'thalach': np.random.randint(71, 203, n),
        'exang': np.random.randint(0, 2, n),
        'oldpeak': np.random.uniform(0, 6.2, n),
        'slope': np.random.randint(0, 3, n),
        'ca': np.random.randint(0, 5, n),
        'thal': np.random.randint(0, 4, n),
        'target': np.random.randint(0, 2, n)
    }
    df = pd.DataFrame(data)
    df.to_csv("heart_disease_proxy.csv", index=False)
    print("Generated heart_disease_proxy.csv")

if __name__ == "__main__":
    generate_heart_disease_proxy()
