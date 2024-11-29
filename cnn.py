import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
import joblib
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import cv2

class EmotionClassifier:
    def __init__(self, img_size=64, num_classes=7):
        self.img_size = img_size
        self.num_classes = num_classes
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    
    def load_images(self, dataset_path):
        X_train, y_train, X_test, y_test = [], [], [], []
        
        for split in ['Train', 'Test']:
            data_path = os.path.join(dataset_path, split)
            for label_idx, label in enumerate(self.emotions):
                folder_path = os.path.join(data_path, label)
                for img_name in os.listdir(folder_path):
                    img_path = os.path.join(folder_path, img_name)
                    img = load_img(img_path, color_mode='grayscale', target_size=(self.img_size, self.img_size))
                    
                    # Enhanced preprocessing
                    img_array = img_to_array(img)
                    img_array = cv2.equalizeHist(img_array.astype(np.uint8))  # Histogram equalization
                    img_array = img_array / 127.5 - 1.0  # Normalize to [-1, 1]
                    
                    if split == 'Train':
                        X_train.append(img_array)
                        y_train.append(label_idx)
                    else:
                        X_test.append(img_array)
                        y_test.append(label_idx)
        
        return (np.array(X_train)[..., np.newaxis], 
                np.array(X_test)[..., np.newaxis], 
                to_categorical(y_train), 
                to_categorical(y_test))

    from tensorflow.keras.layers import Input

    def build_advanced_cnn(self):
        input_tensor = Input(shape=(self.img_size, self.img_size, 1))  # Define input shape
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_tensor)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.25)(x)

        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.25)(x)

        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.25)(x)

        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        output_tensor = Dense(self.num_classes, activation='softmax')(x)

        model = Model(inputs=input_tensor, outputs=output_tensor)
        model.compile(optimizer=Adam(learning_rate=0.1),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
        return model


    def train_model(self, X_train, y_train, X_test, y_test):
        # Advanced data augmentation
        datagen = ImageDataGenerator(
            rotation_range=20, 
            width_shift_range=0.2, 
            height_shift_range=0.2, 
            horizontal_flip=True, 
            brightness_range=[0.8, 1.2], 
            zoom_range=0.2, 
            fill_mode='nearest'
        )
        
        model = self.build_advanced_cnn()
        
        # Callbacks
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.1)
        early_stop = EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True)
        
        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=64), 
            validation_data=(X_test, y_test), 
            epochs=100, 
            callbacks=[early_stop]
        )
        
        return model, history

    def extract_cnn_features(self, model, X):
        # Build the feature extractor after calling the model
        _ = model.predict(X[:1])  # Call the model with a dummy batch to initialize
        feature_extractor = Model(inputs=model.input, outputs=model.layers[-2].output)
        return feature_extractor.predict(X)

    def train_svm(self, X_train_features, y_train):
        # GridSearchCV for hyperparameter tuning
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.1, 0.01],
            'kernel': ['rbf', 'poly']
        }
        
        svm = SVC()
        grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train_features, np.argmax(y_train, axis=1))
        
        return grid_search.best_estimator_

    def evaluate_model(self, svm_classifier, X_test_features, y_test):
        y_pred = svm_classifier.predict(X_test_features)
        accuracy = np.mean(y_pred == np.argmax(y_test, axis=1))
        
        print(f'Hybrid Model Accuracy: {accuracy:.2f}')
        print('\nConfusion Matrix:')
        print(confusion_matrix(np.argmax(y_test, axis=1), y_pred))
        print('\nClassification Report:')
        print(classification_report(np.argmax(y_test, axis=1), y_pred, target_names=self.emotions))

def main():
    classifier = EmotionClassifier()
    X_train, X_test, y_train, y_test = classifier.load_images('./DATASET')
    
    # Train CNN
    cnn_model, _ = classifier.train_model(X_train, y_train, X_test, y_test)
    
    # Extract CNN features
    X_train_features = classifier.extract_cnn_features(cnn_model, X_train)
    X_test_features = classifier.extract_cnn_features(cnn_model, X_test)
    
    # Train SVM on CNN features
    svm_classifier = classifier.train_svm(X_train_features, y_train)
    
    # Evaluate
    classifier.evaluate_model(svm_classifier, X_test_features, y_test)
    
    cnn_model.save("cnn_model.h5")
    joblib.dump(svm_classifier, "svm_cnn_model.joblib")

if __name__ == '__main__':
    main()
