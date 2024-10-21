import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC, SVC, NuSVC
from sklearn.linear_model import SGDClassifier, LogisticRegression, LogisticRegressionCV, Perceptron, PassiveAggressiveClassifier, RidgeClassifier, RidgeClassifierCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.neural_network import MLPClassifier
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageEnhance, ImageOps
import os
from sklearn.preprocessing import RobustScaler

def prepare_data(train_images, test_images):
    X_train, y_train, X_test, y_test = [], [], [], []

    for label, images in train_images.items():
        for img in images:
            X_train.append(img.flatten())
            y_train.append(label)

    for label, images in test_images.items():
        for img in images:
            X_test.append(img.flatten())
            y_test.append(label)

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, y_train, X_test_scaled, y_test


def apply_dimension_reduction(X_train, y_train, X_test, method, n_components):
    if method == 'PCA':
        reducer = PCA(n_components=n_components)
        X_train_reduced = reducer.fit_transform(X_train)
        X_test_reduced = reducer.transform(X_test)
    elif method in ['FDA', 'LDA']:
        reducer = LinearDiscriminantAnalysis(n_components=n_components)
        X_train_reduced = reducer.fit_transform(X_train, y_train)
        X_test_reduced = reducer.transform(X_test)
    else:
        raise ValueError("Invalid dimension reduction method")

    return X_train_reduced, X_test_reduced

def get_classifiers():
    return {
        'LinearSVC': LinearSVC(random_state=42, max_iter=2000),
        'SGDClassifier': SGDClassifier(random_state=42, max_iter=2000, tol=1e-3),
        'MLPClassifier': MLPClassifier(random_state=42, max_iter=1000, early_stopping=True),
        'Perceptron': Perceptron(random_state=42, max_iter=2000),
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000, solver='lbfgs'),
        'LogisticRegressionCV': LogisticRegressionCV(random_state=42, max_iter=1000, cv=5),
        'SVC': SVC(random_state=42, max_iter=2000),
        'CalibratedClassifierCV': CalibratedClassifierCV(estimator=LinearSVC(random_state=42, max_iter=2000)),
        'PassiveAggressiveClassifier': PassiveAggressiveClassifier(random_state=42, max_iter=2000),
        'LabelPropagation': LabelPropagation(max_iter=2000),
        'LabelSpreading': LabelSpreading(max_iter=2000),
        'RandomForestClassifier': RandomForestClassifier(random_state=42, n_estimators=100),
        'GradientBoostingClassifier': GradientBoostingClassifier(random_state=42, n_estimators=100),
        'QuadraticDiscriminantAnalysis': QuadraticDiscriminantAnalysis(),
        'HistGradientBoostingClassifier': HistGradientBoostingClassifier(random_state=42, max_iter=200),
        'RidgeClassifier': RidgeClassifier(random_state=42, max_iter=2000),
        'RidgeClassifierCV': RidgeClassifierCV(cv=5),
        'AdaBoostClassifier': AdaBoostClassifier(random_state=42, n_estimators=100),
        'ExtraTreesClassifier': ExtraTreesClassifier(random_state=42, n_estimators=100),
        'KNeighborsClassifier': KNeighborsClassifier(n_neighbors=5),
        'BaggingClassifier': BaggingClassifier(random_state=42, n_estimators=10),
        'BernoulliNB': BernoulliNB(),
        'LinearDiscriminantAnalysis': LinearDiscriminantAnalysis(),
        'GaussianNB': GaussianNB(),
        'NuSVC': NuSVC(random_state=42, max_iter=2000),
        'DecisionTreeClassifier': DecisionTreeClassifier(random_state=42),
        'NearestCentroid': NearestCentroid(),
        'ExtraTreeClassifier': ExtraTreeClassifier(random_state=42),
    }


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compare_classifiers_with_dimension_reduction(train_images, test_images):
    X_train, y_train, X_test, y_test = prepare_data(train_images, test_images)
    
    dimension_reduction_methods = {
        'PCA': 50,
        'FDA': 1,
        'LDA': 6
    }
    
    classifiers = get_classifiers()
    results = np.zeros((len(classifiers), len(dimension_reduction_methods)))
    
    for i, (dr_method, n_components) in enumerate(dimension_reduction_methods.items()):
        X_train_reduced, X_test_reduced = apply_dimension_reduction(X_train, y_train, X_test, dr_method, n_components)
        
        for j, (name, clf) in enumerate(classifiers.items()):
            try:
                clf.fit(X_train_reduced, y_train)
                y_pred = clf.predict(X_test_reduced)
                accuracy = accuracy_score(y_test, y_pred)
                results[j, i] = accuracy
                logger.info(f"{dr_method} - {name}: Accuracy = {accuracy:.4f}")
            except Exception as e:
                logger.error(f"{dr_method} - {name}: Failed due to {str(e)}")
                results[j, i] = np.nan
    
    return results, list(classifiers.keys()), list(dimension_reduction_methods.keys())





# Define paths to the train and test directories
# train_dir = 'Dataset/Train'
# test_dir = 'Dataset/Test'

train_dir = 'train'
test_dir = 'test'

# Check if the directories exist
if not os.path.exists(train_dir):
    raise FileNotFoundError(f"Training directory {train_dir} not found.")
if not os.path.exists(test_dir):
    raise FileNotFoundError(f"Test directory {test_dir} not found.")

# Function to load images from a directory and its subdirectories
def load_images_from_directory(directory):
    images = {}
    for root, _, files in os.walk(directory):
        class_name = os.path.basename(root)
        images[class_name] = []
        for filename in files:
            if filename.endswith(".jpg") or filename.endswith(".png"):
                img_path = os.path.join(root, filename)
                with Image.open(img_path) as img:
                    img_array = np.array(img)
                    images[class_name].append(img_array)
    return images

# Load images from the train and test directories
train_images = load_images_from_directory(train_dir)
test_images = load_images_from_directory(test_dir)


# Run the comparison
results, classifier_names, dr_methods = compare_classifiers_with_dimension_reduction(train_images, test_images)

# Print the results matrix
print("\nAccuracy Matrix (Classifiers x Dimension Reduction Methods):")
print("Classifier Name".ljust(30) + "\t".join(dr_methods))
for name, row in zip(classifier_names, results):
    print(f"{name.ljust(30)}{row[0]:.4f}\t{row[1]:.4f}\t{row[2]:.4f}")

# Optional: Create a heatmap of the results
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 10))
sns.heatmap(results, annot=True, fmt='.4f', xticklabels=dr_methods, yticklabels=classifier_names, cmap='YlGnBu')
plt.title('Classifier Accuracy with Different Dimension Reduction Methods')
plt.xlabel('Dimension Reduction Method')
plt.ylabel('Classifier')
plt.tight_layout()
plt.show()