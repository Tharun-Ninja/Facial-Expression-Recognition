from sklearn.mixture import GaussianMixture

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from tensorflow.keras.preprocessing.image import load_img, img_to_array


import joblib  # For saving and loading models

class FacialEmotionClassifier:
    def __init__(self, input_shape=(48, 48, 1), num_classes=7):
        """
        Initialize the Facial Emotion Classifier
        
        Args:
            input_shape (tuple): Shape of input images
            num_classes (int): Number of emotion classes
        """
        print("Initializing Facial Emotion Classifier...")
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.img_size = 48
        
        print("Building CNN Feature Extractor...")
        self.cnn_feature_extractor = self._build_cnn_feature_extractor()
        self.scaler = None
        self.gmm = None


    def save(self, model_path="facial_emotion_classifier"):
        """
        Save the trained GMM and scaler to disk
        
        Args:
            model_path (str): Base path to save the model files
        """
        print(f"Saving model to '{model_path}'...")
        if self.gmm is None or self.scaler is None:
            print("Error: Model has not been trained. Save operation aborted.")
            return
        
        joblib.dump(self.gmm, f"{model_path}_gmm.pkl")
        joblib.dump(self.scaler, f"{model_path}_scaler.pkl")
        self.cnn_feature_extractor.save(f"{model_path}_cnn.h5")
        print("Model saved successfully.")

    def load(self, model_path="facial_emotion_classifier"):
        """
        Load the GMM, scaler, and CNN from disk
        
        Args:
            model_path (str): Base path to load the model files from
        """
        print(f"Loading model from '{model_path}'...")
        try:
            self.gmm = joblib.load(f"{model_path}_gmm.pkl")
            self.scaler = joblib.load(f"{model_path}_scaler.pkl")
            self.cnn_feature_extractor = tf.keras.models.load_model(f"{model_path}_cnn.h5")
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")

    def _build_cnn_feature_extractor(self):
        """
        Build a CNN model for feature extraction
        
        Returns:
            tf.keras.Model: CNN feature extractor
        """
        model = models.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Flatten and feature extraction
            layers.Flatten()
        ])
        
        return model
    
    def extract_features(self, X, batch_size=64):
        """
        Extract features using the CNN in batches
        
        Args:
            X (numpy.ndarray): Input images
            batch_size (int): Batch size for prediction
        
        Returns:
            numpy.ndarray: Extracted features
        """
        print(f"Extracting features for {len(X)} images in batches of {batch_size}...")
        num_batches = len(X) // batch_size + int(len(X) % batch_size != 0)
        features_list = []
        for i in range(num_batches):
            batch = X[i * batch_size:(i + 1) * batch_size]
            batch_features = self.cnn_feature_extractor.predict(batch, verbose=0)
            features_list.append(batch_features)
        return np.vstack(features_list)



    def train(self, X_train, y_train):
        """
        Train the classifier by extracting features and training GMM
        
        Args:
            X_train (numpy.ndarray): Training images
            y_train (numpy.ndarray): Training labels
        """
        print("Starting model training...")

        # Extract features
        start_time = time.time()
        X_train_features = self.extract_features(X_train)
        feature_extract_time = time.time() - start_time
        print(f"Total feature extraction time: {feature_extract_time:.2f} seconds")

        # Scale features
        print("Scaling features...")
        start_time = time.time()
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train_features)
        scale_time = time.time() - start_time
        print(f"Feature scaling completed in {scale_time:.2f} seconds")

        # Train GMM
        print("Training Gaussian Mixture Model...")
        start_time = time.time()
        self.gmm = GaussianMixture(n_components=self.num_classes, covariance_type='full', random_state=42)
        self.gmm.fit(X_train_scaled)
        train_time = time.time() - start_time
        print(f"GMM training completed in {train_time:.2f} seconds")

        # Map GMM clusters to the actual labels
        gmm_labels = self.gmm.predict(X_train_scaled)
        print("\nTraining Classification Report:")
        print(classification_report(y_train, gmm_labels, target_names=self.emotions))

    def predict(self, X):
        """
        Predict emotions for input images using GMM
        
        Args:
            X (numpy.ndarray): Input images
        
        Returns:
            numpy.ndarray: Predicted emotion labels
        """
        print(f"Predicting emotions for {len(X)} images...")
        X_features = self.extract_features(X)
        X_scaled = self.scaler.transform(X_features)
        predictions = self.gmm.predict(X_scaled)
        print("Prediction complete.")
        return predictions
    
    
    def plot_confusion_matrix(self, X_test, y_test):
        """
        Plot confusion matrix for model evaluation
        
        Args:
            X_test (numpy.ndarray): Test images
            y_test (numpy.ndarray): Test labels
        """
        print("Generating Confusion Matrix...")
        X_test_features = self.extract_features(X_test)
        X_test_scaled = self.scaler.transform(X_test_features)
        y_pred = self.gmm.predict(X_test_scaled)
        
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.emotions, 
                    yticklabels=self.emotions)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.show()


def load_example_data():
    """
    Load data from image folders
    
    Returns:
        Tuple of training and test datasets
    """
    print("Loading image data...")
    X_train, y_train = [], []
    X_test, y_test = [], []
    
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    
    for split in ['Train', 'Test']:
        data_path = os.path.join("/content/drive/MyDrive/DUMPY/Dataset", split)
        print(f"Processing {split} dataset...")
        
        for label_idx, label in enumerate(emotions):
            folder_path = os.path.join(data_path, label.lower())
            
            # Ensure folder exists
            if not os.path.exists(folder_path):
                print(f"Warning: Folder {folder_path} does not exist.")
                continue
            
            for img_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_name)
                
                try:
                    # Load image in grayscale
                    img = load_img(img_path, color_mode='grayscale', target_size=(48, 48))
                    img_array = img_to_array(img) / 255.0  # Normalize
                    
                    if split == 'Train':
                        X_train.append(img_array)
                        y_train.append(label_idx)
                    else:
                        X_test.append(img_array)
                        y_test.append(label_idx)
                
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")

    # Convert to numpy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    print(f"Training dataset shape: {X_train.shape}")
    print(f"Test dataset shape: {X_test.shape}")
    
    return X_train, y_train, X_test, y_test


def compute_accuracy(y_true, y_pred):
    """
    Compute and display accuracy.
    
    Args:
        y_true (numpy.ndarray): True labels
        y_pred (numpy.ndarray): Predicted labels
    
    Returns:
        float: Accuracy
    """
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.2%}")
    return accuracy

import pickle

def save_data(X_train, y_train, X_test, y_test, filename="dataset.pkl"):
    """
    Save the datasets to a file using pickle.
    
    Args:
        X_train (np.array): Training features
        y_train (np.array): Training labels
        X_test (np.array): Test features
        y_test (np.array): Test labels
        filename (str): Path to save the file
    """
    print(f"Saving datasets to {filename}...")
    with open(filename, "wb") as file:
        pickle.dump((X_train, y_train, X_test, y_test), file)
    print("Datasets saved successfully.")

def load_data(filename="dataset.pkl"):
    """
    Load the datasets from a file using pickle.
    
    Args:
        filename (str): Path to the saved file
    
    Returns:
        tuple: Loaded datasets (X_train, y_train, X_test, y_test)
    """
    print(f"Loading datasets from {filename}...")
    with open(filename, "rb") as file:
        X_train, y_train, X_test, y_test = pickle.load(file)
    print("Datasets loaded successfully.")
    return X_train, y_train, X_test, y_test




print("Starting Facial Emotion Classification...")
start_total_time = time.time()

# Load data
print("Loading data...")
X_train, y_train, X_test, y_test = load_example_data()

# Assuming X_train, y_train, X_test, y_test are already loaded
save_data(X_train, y_train, X_test, y_test, filename="/content/drive/MyDrive/DUMPY/Dataset/emotion_dataset.pkl")


# Load the datasets
X_train, y_train, X_test, y_test = load_data(filename="/content/drive/MyDrive/DUMPY/Dataset/emotion_dataset.pkl")

# Verify the shapes
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

# Initialize and train classifier
print("\nInitializing Classifier...")
classifier = FacialEmotionClassifier()

# Train the model
print("\nTraining Model...")
classifier.train(X_train, y_train)

# Evaluate and plot confusion matrix
print("\nEvaluating on Test Set...")
test_pred = classifier.predict(X_test)

# Save
classifier.save("/content/drive/MyDrive/DUMPY/Dataset/facial_emotion_classifier")


print("\nTest Set Classification Report:")
print(classification_report(y_test, test_pred, target_names=classifier.emotions))

# Calculate accuracy
compute_accuracy(y_test, test_pred)

# Plot confusion matrix
classifier.plot_confusion_matrix(X_test, y_test)

# Total execution time
total_time = time.time() - start_total_time
print(f"\nTotal Execution Time: {total_time:.2f} seconds")



import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from sklearn.mixture import GaussianMixture
import tensorflow as tf
from tensorflow.keras import layers, models

def build_cnn_feature_extractor(input_shape):
    """
    Build a CNN model for feature extraction.

    Args:
        input_shape (tuple): Shape of input images.

    Returns:
        tf.keras.Model: CNN feature extractor.
    """
    model = models.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Flatten and feature extraction
        layers.Flatten()
    ])
    return model

def fit_gmm_no_scaling(cnn_feature_extractor, gmm_path, X_train, y_train, X_test, y_test, emotions):
    """
    Fit a Gaussian Mixture Model (GMM) on features extracted from CNN without scaling.

    Args:
        cnn_feature_extractor (tf.keras.Model): Pretrained CNN feature extractor.
        gmm_path (str): Path to save the GMM model.
        X_train (numpy.ndarray): Training images.
        y_train (numpy.ndarray): Training labels.
        X_test (numpy.ndarray): Test images.
        y_test (numpy.ndarray): Test labels.
        emotions (list): List of emotion labels.

    Returns:
        None
    """
    print("Extracting features from training data...")
    X_train_features = cnn_feature_extractor.predict(X_train, verbose=0)

    print("Training Gaussian Mixture Model without scaling...")
    gmm = GaussianMixture(n_components=len(emotions), covariance_type='full', random_state=42)
    gmm.fit(X_train_features)

    print("Extracting features from test data...")
    X_test_features = cnn_feature_extractor.predict(X_test, verbose=0)

    print("Predicting on test data...")
    y_pred = gmm.predict(X_test_features)

    print("\nTest Set Classification Report:")
    print(classification_report(y_test, y_pred, target_names=emotions))

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy on Test Data (without scaling): {accuracy:.2%}")

    # Save the GMM model
    print(f"Saving GMM model to {gmm_path}...")
    joblib.dump(gmm, gmm_path)
    print("GMM model saved successfully.")

# Example usage
input_shape = (48, 48, 1)
cnn_feature_extractor = build_cnn_feature_extractor(input_shape)
classifier = FacialEmotionClassifier()
classifier.load("/content/drive/MyDrive/DUMPY/Dataset/facial_emotion_classifier_cnn.h5")

fit_gmm_no_scaling(
    cnn_feature_extractor=classifier.cnn_feature_extractor,
    gmm_path="/content/drive/MyDrive/DUMPY/Dataset/facial_emotion_gmm_noscale.pkl",
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    emotions=classifier.emotions
)
