print("🚀 Starting Project...")
import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D

# ⚙️ Config
DATA_DIR = "PetImages"
CLASSES = ["Cat", "Dog"]
IMAGE_SIZE = 224          # Best size for VGG16
MAX_IMAGES = 2000         # Limit to avoid memory issues
dataset = []

# image loading 
def load_dataset():
    print("\n📂 Loading images...")
    for class_name in CLASSES:
        folder_path = os.path.join(DATA_DIR, class_name)
        label = CLASSES.index(class_name)
        print(f"\n👉 Reading {class_name} images...")
        count = 0
        for file in tqdm(os.listdir(folder_path)):
              if count >= MAX_IMAGES:
                break
            try:
                img_path = os.path.join(folder_path, file)
                image = cv2.imread(img_path)
                if image is None:
                    continue
                # Resize image
                image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
                if image.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
                    continue
                dataset.append((image, label))
                count += 1
            except Exception:
                continue
    print(f"\n✅ Total images loaded: {len(dataset)}")

# Run data loading
load_dataset()

# 🔄data preparation
print("\n🔄 Preparing data...")
X, y = [], []
for image, label in dataset:
    X.append(image)
    y.append(label)
X = np.array(X, dtype="float32")
y = np.array(y)

#   VGG16 for process
X = preprocess_input(X)
print("✅ Data ready!")

# 🧠 feature extraction
print("\n🧠 Extracting deep features using VGG16...")

# loading pretrained data
base_model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
)

# Freeze layers 
for layer in base_model.layers:
    layer.trainable = False

# reduce feature size
x = base_model.output
x = GlobalAveragePooling2D()(x)
feature_extractor = Model(inputs=base_model.input, outputs=x)

# Extract features
features = feature_extractor.predict(X, batch_size=32, verbose=1)
print("✔ Feature shape:", features.shape)


# 🔀 data shuffling 
features, y = shuffle(features, y, random_state=42)

# ✂️ split train data
X_train, X_test, y_train, y_test = train_test_split(
    features, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print("✅ Data split completed")


# 📏 Feature Scaling
print("\n📏 Scaling features...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ⚙️ training SVM
print("\n⚙️ Training SVM classifier...")
svm_model = SVC(
    kernel='rbf',
    C=50,
    gamma=0.0005
)
svm_model.fit(X_train, y_train)
print("✅ Training completed!")


# 📊  Evaluate Model
print("\n📊 Evaluating model...")
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 Accuracy: {accuracy:.2f}")
print("\n📋 Classification Report:\n")
print(classification_report(y_test, y_pred))
print("\n🎉 Project Completed Successfully!")