# SkillCraft Technology – Task 03: SVM Classifier for Cats vs Dogs

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# ⚠️ Set your dataset path
# Make sure you unzip the Kaggle cats and dogs dataset and update this path accordingly
DATADIR = "dataset"  # replace with actual path
CATEGORIES = ["Dog", "Cat"]

data = []
labels = []

# Load and preprocess images (resize to 64x64 for faster processing)
for category in CATEGORIES:
    path = os.path.join(DATADIR, category)
    class_num = CATEGORIES.index(category)
    for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            resized_array = cv2.resize(img_array, (64, 64))  # Resize
            data.append(resized_array.flatten())  # Flatten to 1D
            labels.append(class_num)
        except Exception as e:
            pass

# Convert to NumPy arrays
X = np.array(data)
y = np.array(labels)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM model
model = SVC(kernel='linear')  # or 'rbf'
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
