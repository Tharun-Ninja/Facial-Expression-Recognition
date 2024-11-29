import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Dense, Dropout, GlobalAveragePooling2D,
                                     BatchNormalization)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

def gamma_transform(features, gamma=0.5):
    """
    Apply gamma transformation for feature normalization.

    Args:
        features (ndarray): Input features to normalize.
        gamma (float): Gamma value for the transformation.

    Returns:
        ndarray: Transformed features.
    """
    return np.power(features, gamma)

def get_subfolders_and_image_counts(directory):
    """
    Get subfolder names and their image counts in the specified directory.

    Args:
        directory (str): Path to the dataset directory.

    Returns:
        dict: Dictionary with folder names as keys and image counts as values.
    """
    subfolders = {}
    for subfolder in os.listdir(directory):
        subfolder_path = os.path.join(directory, subfolder)
        if os.path.isdir(subfolder_path):
            subfolders[subfolder] = len(os.listdir(subfolder_path))
    return subfolders

# Set dataset paths
train_dir = '/kaggle/input/train'
test_dir = '/kaggle/input/test'

# Display folder and image counts
train_subfolders = get_subfolders_and_image_counts(train_dir)
test_subfolders = get_subfolders_and_image_counts(test_dir)

print("Training Subfolders and Image Counts:")
for folder, count in train_subfolders.items():
    print(f"{folder}: {count} images")

print("\nTest Subfolders and Image Counts:")
for folder, count in test_subfolders.items():
    print(f"{folder}: {count} images")

# Visualize sample images from the training dataset
folders = [folder for folder in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, folder))]
for folder in folders:
    folder_path = os.path.join(train_dir, folder)
    image_files = os.listdir(folder_path)[:5]

    fig, axes = plt.subplots(1, 5, figsize=(15, 5))
    fig.suptitle(f" Class: {folder}", fontsize=16)

    for i, image_file in enumerate(image_files):
        image_path = os.path.join(folder_path, image_file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        axes[i].imshow(image, cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f"Image {i+1}")

    plt.show()

# Prepare file paths and labels for the dataset
filepaths, labels = [], []

folders = os.listdir(train_dir)
for folder in folders:
    folder_path = os.path.join(train_dir, folder)
    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            filepaths.append(os.path.join(folder_path, file))
            labels.append(folder)

# Create a DataFrame for the training dataset
Fseries = pd.Series(filepaths, name='filepaths')
Lseries = pd.Series(labels, name='labels')
train_df = pd.concat([Fseries, Lseries], axis=1)

# Data augmentation
train_datagen = ImageDataGenerator(rescale=1./255,
                                    rotation_range=20,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    validation_split=0.2)

train_gen = train_datagen.flow_from_dataframe(
    train_df,
    x_col='filepaths',
    y_col='labels',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

valid_gen = train_datagen.flow_from_dataframe(
    train_df,
    x_col='filepaths',
    y_col='labels',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Define the CNN model with ResNet50 as the base
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)

x = Dense(1024, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)

x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)

x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)

# Modify the final Dense layer to output 15 neurons
emotion_output = Dense(15, activation='softmax', name='emotion_output')(x)

# Create the model
model = Model(inputs=base_model.input, outputs=emotion_output)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Display the model summary
model.summary()

# Optionally apply gamma transformation to model outputs
# def process_model_output(model_output):
#     return gamma_transform(model_output)


from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Model

# Extract features using the CNN
feature_extractor_model = Model(inputs=model.input, outputs=model.get_layer('emotion_output').output)

# Generate features for training data
train_features = []
train_labels = []

for batch in train_gen:
    features = feature_extractor_model.predict(batch[0])
    train_features.append(features)
    train_labels.extend(batch[1])
    if len(train_features) * train_gen.batch_size >= train_gen.n:
        break

train_features = np.vstack(train_features)
train_labels = np.argmax(np.vstack(train_labels), axis=1)  # Convert one-hot to integer labels

# Train the Gaussian Mixture Model (GMM)
gmm = GaussianMixture(n_components=15, random_state=42)
gmm.fit(train_features)

# Predict labels for training data
gmm_predictions = gmm.predict(train_features)

# Calculate accuracy
gmm_accuracy = accuracy_score(train_labels, gmm_predictions)
print(f"GMM Accuracy on Training Data: {gmm_accuracy:.4f}")

# Similarly, process the validation data for accuracy (optional)
val_features = []
val_labels = []

for batch in valid_gen:
    features = feature_extractor_model.predict(batch[0])
    val_features.append(features)
    val_labels.extend(batch[1])
    if len(val_features) * valid_gen.batch_size >= valid_gen.n:
        break

val_features = np.vstack(val_features)
val_labels = np.argmax(np.vstack(val_labels), axis=1)

val_predictions = gmm.predict(val_features)
val_accuracy = accuracy_score(val_labels, val_predictions)
print(f"GMM Accuracy on Validation Data: {val_accuracy:.4f}")
