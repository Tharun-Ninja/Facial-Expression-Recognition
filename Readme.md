# Facial Expression Recognition

This project implements a facial expression recognition system using various machine learning and deep learning techniques. The goal is to classify facial expressions into one of the seven emotions: angry, disgust, fear, happy, neutral, sad, and surprise.

---

## Getting Started

### Requirements
- Python 3.x
- Git
- Jupyter Notebook

---

## Installation and Setup

### Clone the Repository

First, clone the repository to your local machine:

```bash
git clone https://github.com/Tharun-Ninja/facial-expression-recognition.git
cd facial-expression-recognition
```

### Create a Virtual Environment

Create a virtual environment to manage your project dependencies:

```bash
python -m venv venv

# Activate the virtual environment
# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### Install the Required Libraries

Once the virtual environment is activated, install the required packages:

```bash
pip install -r requirements.txt
```

### Run the Jupyter Notebook

Open and run the `tharun.ipynb` file in Jupyter Notebook. If you encounter the following error:

```css
TypeError: OneHotEncoder.__init__() got an unexpected keyword argument 'sparse'
```

Fix it by navigating to the following file:

```bash
code .ml_env/lib/python3.12/site-packages/lazypredict/Supervised.py
```

Change line 98 from:

```python
("encoding", OneHotEncoder(handle_unknown="ignore", sparse=False)),
```

to:

```python
("encoding", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
```

---

## Usage

Follow the instructions within the notebook to:
- Load the dataset.
- Preprocess the images.
- Extract features.
- Train the model.
- Evaluate its performance.

---

## Implementation Details

### Dataset
- **FER2013 Dataset**: Contains 35,887 grayscale images (48x48 pixels) labeled with seven emotions.
- **Preprocessing**: Includes resizing, normalization, and data augmentation (flipping, rotation, Gaussian noise).

### Exploratory Data Analysis (EDA)
- **Pixel Intensity Distribution**: Analyzed pixel intensity distributions.
- **Correlation Heatmap**: Identified relationships among pixel intensities.
- **PCA**: Applied Principal Component Analysis for dimensionality reduction.
- **Normalization**: Scaled images to the range [0, 1].
- **Class Label Indexing**: Assigned integer labels to emotion classes.

### Preprocessing
- Data Augmentation Techniques:
  - Random rotation (-30° to 30°)
  - Horizontal and vertical flips
  - Gaussian noise

### Model Architectures

#### CNN (Convolutional Neural Network)
- **Input Layer**: 48x48 grayscale images.
- **Convolutional Blocks**:
  - Block 1: Conv2D (32 filters, 3x3), BatchNormalization, MaxPooling2D, Dropout (0.25).
  - Block 2: Conv2D (64 filters, 3x3), BatchNormalization, MaxPooling2D, Dropout (0.25).
  - Block 3: Conv2D (128 filters, 3x3), BatchNormalization, MaxPooling2D, Dropout (0.25).
- **Output Layer**: Flattened features for classification.

#### CNN + SVM Classifier
- Features extracted using CNN were scaled with StandardScaler.
- Trained a Support Vector Machine (SVM) classifier on the scaled features.

#### Random Forest Classifier
- Extracted histogram of oriented gradients (HOG) features from images.
- Trained a Random Forest Classifier as a baseline model.

#### Gaussian Mixture Model (GMM)
- Clustering approach to group facial expression data into seven clusters.

#### Support Vector Classifier (SVC)
- Applied on HOG features after scaling for standalone classification.

#### Transfer Learning with VGG16
- Utilized the pre-trained VGG16 model (excluding fully connected layers).
- Added custom dense layers for emotion classification.

---

## Training Process
1. **Feature Extraction**: Used respective models to extract features from images.
2. **Feature Scaling**: Normalized extracted features using StandardScaler.
3. **Model Training**: Trained models on processed features.

---

## Evaluation

### Metrics
- **Precision**, **Recall**, **F1-score** for each emotion class.
- Overall **Accuracy** and **Confusion Matrix** to assess performance.

### Results Summary
| Model                    | Accuracy | Weighted F1-Score |
|--------------------------|----------|--------------------|
| CNN                      | 62%      | 0.62               |
| CNN + SVM                | 48%      | 0.46               |

---

## Future Improvements
1. Address class imbalance using advanced augmentation techniques or oversampling.
2. Experiment with hybrid models combining CNN with ensemble classifiers.
3. Incorporate more advanced transfer learning approaches, such as fine-tuning pre-trained models.

---

## Contributing

Feel free to fork the repository, make changes, and create pull requests. Contributions are welcome!

