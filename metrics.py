import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# === STEP 1: Load the model ===
model = load_model("C:\\Users\\yuktanidhi\\Downloads\\Sign-Language-To-Text-and-Speech-Conversion-master\\cnn8grps_rad1_model.h5")
print("\n✅ Model loaded! It expects input shape:", model.input_shape)

# === STEP 2: Prepare dataset ===
data_dir = 'C:\\Users\\yuktanidhi\\Downloads\\Sign-Language-To-Text-and-Speech-Conversion-master\\AtoZ_3.1'
img_size = (400, 400)
num_test_images_per_class = 10  # Increase test images per class for better evaluation

X_test = []
y_test = []

# Mapping class names (folder names A-Z) to numbers
class_names = sorted(os.listdir(data_dir))
class_to_label = {class_name: idx for idx, class_name in enumerate(class_names)}

for class_name in class_names:
    class_folder = os.path.join(data_dir, class_name)
    image_files = os.listdir(class_folder)
    
    # If less images available than requested, adjust
    if len(image_files) < num_test_images_per_class:
        selected_images = image_files
    else:
        selected_images = random.sample(image_files, num_test_images_per_class)
    
    for img_file in selected_images:
        img_path = os.path.join(class_folder, img_file)
        img = image.load_img(img_path, target_size=img_size)
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0  # Normalize
        X_test.append(img_array)
        y_test.append(class_to_label[class_name])

# Convert to numpy arrays
X_test = np.array(X_test)
y_test = np.array(y_test)

print(f"\n✅ Loaded {len(X_test)} images for testing.")

# === STEP 3: Predict ===
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# === STEP 4: Metrics Calculation ===
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

print("\n📊 Performance Metrics:")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# === STEP 5: Confusion Matrix ===
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(16,14))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, 
            yticklabels=class_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# === STEP 6: Detailed Classification Report ===
print("\n📝 Detailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))
