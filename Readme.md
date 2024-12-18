# Facial Expression Recognition Project

This project implements a facial expression recognition system using machine learning techniques. The dataset contains images categorized by different emotions, and the model is trained to recognize these emotions based on image features.

## Getting Started

### Requirements
- Python 3.x
- Git
- Jupyter Notebook

### Installation and Setup

1. **Clone the Repository**

   First, clone the repository to your local machine:

   ```bash
   git clone https://github.com/Tharun-Ninja/facial-expression-recognition.git
   cd facial-expression-recognition
   ```

2. **Create a Virtual Environment**

   Create a virtual environment to manage your project dependencies:
   
   ```bash
   python -m venv venv

   # Activate the virtual environment
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install the Required Libraries**

   Once the virtual environment is activated, install the required packages:
   
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Jupyter Notebook**
    
   Run `tharun.ipynb` file in Jupyter Notebook.

   If you encounter the following error:
   
   ```
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

5. **Usage**

   Follow the instructions within the notebook to load the dataset, preprocess the images, extract features, train the model, and evaluate its performance.

6. **Contributing**

   Feel free to fork the repository, make changes, and create pull requests. Contributions are welcome!

## Implementation Details

### Dataset
- **FER2013 Dataset**: Contains 35,887 grayscale images (48x48 pixels) labeled with seven emotions: angry, disgust, fear, happy, neutral, sad, and surprise. Preprocessing steps include resizing, normalization, and data augmentation (flipping, rotation, Gaussian noise).

### Exploratory Data Analysis (EDA)
1. **Pixel Intensity Distribution**: Analyzed pixel intensity distributions to identify imbalances in brightness and contrast.
2. **Correlation Heatmap**: Generated to identify pixel intensity relationships across classes.
3. **PCA**: Used Principal Component Analysis (PCA) for dimensionality reduction and visualization in 2D.
4. **Normalization**: Images were normalized to the range [0, 1] for consistent comparison.
5. **Class Label Indexing**: Assigned integer labels to each emotion class for efficient processing.

### Preprocessing
- Applied data augmentation techniques such as:
  - Random rotation (-30° to 30°)
  - Horizontal and vertical flips
  - Gaussian noise

### Model Architecture
#### CNN Architecture
1. **Input Layer**: 48x48 grayscale images.
2. **Convolutional Blocks**:
   - **Block 1**: Conv2D (32 filters, 3x3), BatchNormalization, MaxPooling2D, Dropout (0.25).
   - **Block 2**: Conv2D (64 filters, 3x3), BatchNormalization, MaxPooling2D, Dropout (0.25).
   - **Block 3**: Conv2D (128 filters, 3x3), BatchNormalization, MaxPooling2D, Dropout (0.25).
3. **Output Layer**: Flattened features for classification.

#### CNN + SVM Classifier
- Features extracted using CNN were scaled with StandardScaler.
- Trained a Support Vector Machine (SVM) classifier on the scaled features.

### Training Process
1. **Feature Extraction**: Used the CNN model to extract spatial features from images.
2. **Feature Scaling**: Applied StandardScaler to normalize extracted features.
3. **Model Training**: Trained an SVM classifier on the scaled features.

### Evaluation
- Metrics used for evaluation:
  - **Precision, Recall, F1-score** for each emotion class.
  - Overall **Accuracy** and **Confusion Matrix** to assess performance.

#### Results
- CNN achieved an accuracy of **62%** and a weighted F1-score of **0.62**.
- Performance metrics for individual classes showed variability due to class imbalance.

### Future Improvements
- Address class imbalance using advanced augmentation techniques or oversampling.
- Experiment with hybrid models combining CNN with ensemble classifiers for improved accuracy.
- Utilize transfer learning with pre-trained networks for better generalization.
