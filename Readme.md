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

2.	**Create a Virtual Environment**

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
    
    run `tharun.ipynb` file in jupyter notebook

    if you get this `error`
    `TypeError: OneHotEncoder.__init__() got an unexpected keyword argument 'sparse'`
    then go to this file using this command
    ```bash
    code .ml_env/lib/python3.12/site-packages/lazypredict/Supervised.py   
    ```
    and change the line 98 from
    ```python
    ("encoding", OneHotEncoder(handle_unknown="ignore", sparse=False)),
    ```
    to 
    ```python
    ("encoding", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ```

5. **Usage**

    Follow the instructions within the notebook to load the dataset, preprocess the images, extract features, train the model, and evaluate its performance.

6. **Contributing**

    Feel free to fork the repository, make changes, and create pull requests. Contributions are welcome!

7. **License**

    This project is licensed under the MIT License.
