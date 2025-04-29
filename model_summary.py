# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Load the model
model = load_model('cnn8grps_rad1_modell.h5')  # Replace with your file name

# Print the model summary
print("\n📋 Model Summary:\n")
model.summary()

# --- Now evaluate the model ---

# Path to your dataset (same path used during training)
dataset_dir = r"C:\Users\yuktanidhi\Downloads\Sign-Language-To-Text-and-Speech-Conversion-master\AtoZ_3.1"

# Image dimensions
img_width, img_height = 400, 400

# Reload validation data
val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

validation_generator = val_datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=False  # Important: Don't shuffle
)

# Get predictions
Y_pred = model.predict(validation_generator)
y_pred = np.argmax(Y_pred, axis=1)
y_true = validation_generator.classes

# Labels (A-Z folder names)
labels = list(validation_generator.class_indices.keys())

# Calculate and print performance metrics

# Accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"\n✅ Accuracy: {accuracy:.4f}")

# Precision (macro average)
precision = precision_score(y_true, y_pred, average='macro')
print(f"✅ Precision (macro): {precision:.4f}")

# Recall (macro average)
recall = recall_score(y_true, y_pred, average='macro')
print(f"✅ Recall (macro): {recall:.4f}")

# F1 Score (macro average)
f1 = f1_score(y_true, y_pred, average='macro')
print(f"✅ F1 Score (macro): {f1:.4f}")

# Classification Report
print("\n📊 Detailed Classification Report:")
print(classification_report(y_true, y_pred, target_names=labels))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print("\n📊 Confusion Matrix:")
print(cm)

# Plot Confusion Matrix
plt.figure(figsize=(15, 12))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

