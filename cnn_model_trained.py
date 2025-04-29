# Import necessary libraries
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

# Path to your dataset
dataset_dir = r"C:\Users\yuktanidhi\Downloads\Sign-Language-To-Text-and-Speech-Conversion-master\AtoZ_3.1"

# Image dimensions
img_width, img_height = 400, 400

# Number of classes
num_classes = 26

# Create the model
model = Sequential()

# 1st Convolutional Block
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 2nd Convolutional Block
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 3rd Convolutional Block
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 4th Convolutional Block
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the output
model.add(Flatten())

# Fully Connected Layers
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(96, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(64, activation='relu'))

# Final Output Layer
model.add(Dense(num_classes, activation='softmax'))  # 26 classes now!

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Image data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Load training images
train_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

# Load validation images
validation_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=False  # Important! Don't shuffle validation set
)

# Checkpoint to save best model
checkpoint = ModelCheckpoint("best_model_AtoZ.h5", monitor='val_accuracy', save_best_only=True, verbose=1)

# Train the model
history = model.fit(
    train_generator,
    epochs=25,
    validation_data=validation_generator,
    callbacks=[checkpoint]
)

# Save the final model
model.save("cnn8grps_rad1_modell.h5")

print("\n✅ Training completed and model saved as 'cnn8grps_rad1_modell.h5'.")

# ----------------------------------
# 📈 Now evaluate the model:
# ----------------------------------

# Predict on validation set
Y_pred = model.predict(validation_generator)
y_pred = np.argmax(Y_pred, axis=1)
y_true = validation_generator.classes

# Labels (folder names A-Z)
labels = list(validation_generator.class_indices.keys())

# Classification Report
print("\n📊 Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=labels))

# Confusion Matrix
print("\n📊 Confusion Matrix:\n")
cm = confusion_matrix(y_true, y_pred)

# Plot confusion matrix
plt.figure(figsize=(15, 12))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
