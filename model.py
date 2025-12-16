# save_model.py
import cv2, os, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

def extract_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.resize(img, (100, 100))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mean_r, mean_g, mean_b = np.mean(img_rgb[:,:,0]), np.mean(img_rgb[:,:,1]), np.mean(img_rgb[:,:,2])
    std_r, std_g, std_b = np.std(img_rgb[:,:,0]), np.std(img_rgb[:,:,1]), np.std(img_rgb[:,:,2])
    brightness = np.mean(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY))
    return [mean_r, mean_g, mean_b, std_r, std_g, std_b, brightness]

data, labels = [], []
dataset_path = "C:\Project\AI water\water_quality_model"
classes = ["Good", "Moderate", "Poor"]

for label in classes:
    folder = os.path.join(dataset_path, label)
    for f in os.listdir(folder):
        path = os.path.join(folder, f)
        feat = extract_features(path)
        if feat:
            data.append(feat)
            labels.append(label)

X, y = pd.DataFrame(data), pd.Series(labels)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

joblib.dump(model, "model.pkl")
print("✅ Model trained and saved as model.pkl")
