import os
import json

def create_ml_project_structure(project_dir):
    """
    Creates a predefined ML project folder structure with template files.
    """
    # Define folder structure
    folders = [
        'data/raw',
        'data/processed',
        'notebooks',
        'scripts',
        'models',
        'config',
        'reports/figures',
        'utils'
    ]
    
    # Create folders
    for folder in folders:
        folder_path = os.path.join(project_dir, folder)
        os.makedirs(folder_path, exist_ok=True)
        print(f"Created folder: {folder_path}")
    
    # Create template files
    create_template_files(project_dir)

def create_template_files(project_dir):
    """
    Creates template files inside the ML project structure.
    """
    # Create a requirements.txt file
    requirements_path = os.path.join(project_dir, 'requirements.txt')
    with open(requirements_path, 'w') as f:
        f.write('numpy\npandas\nscikit-learn\nmatplotlib\nseaborn\n')
    print("Created requirements.txt")

    # Create a README.md file
    readme_path = os.path.join(project_dir, 'README.md')
    with open(readme_path, 'w') as f:
        f.write('# ML Project\nThis is a template for ML projects.')
    print("Created README.md")

    # Create a config.yaml file
    config_path = os.path.join(project_dir, 'config/config.yaml')
    with open(config_path, 'w') as f:
        f.write('# Configuration file\n')
    print("Created config.yaml")

    # Create a template notebook

    notebook_path = os.path.join(project_dir, 'notebooks/eda.ipynb')
    # Define the content of the notebook (a markdown cell with EDA title)
    notebook_content = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["# Exploratory Data Analysis\n",
                       "In this notebook, we will cover the first few steps of EDA when we first get a dataset."]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 1. Loading the Data\n",
                       "We start by loading the data into a DataFrame. In Python, this is typically done using `pandas`."]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": ["import pandas as pd\n",
                       "df = pd.read_csv('path_to_data.csv')"]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 2. Checking the First Few Rows\n",
                       "It's important to take a quick look at the data to understand its structure."]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": ["df.head()  # Shows the first 5 rows"]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 3. Summary Statistics\n",
                       "Summary statistics help us understand the distribution and central tendencies of the data."]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": ["df.describe()  # Summary of numeric columns"]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 4. Missing Data Check\n",
                       "We need to check if there are any missing values in our dataset."]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": ["df.isnull().sum()  # Count of missing values in each column"]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 5. Data Types and Basic Info\n",
                       "Check the data types of the columns and the shape of the dataset."]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": ["df.info()  # General info about the DataFrame"]
        }
    ],
    "metadata": {},
    "nbformat": 4,
    "nbformat_minor": 5
    }
    with open(notebook_path, 'w') as f:
        json.dump(notebook_content, f)
    print("Created EDA notebook (eda.ipynb)")

    # Create script templates
    scripts = {
        'scripts/data_preprocessing.py': '''import pandas as pd\n
# Data preprocessing script
def preprocess_data(data):
    # Add your preprocessing steps here
    return data\n
''',
        'scripts/model_training.py': '''#Adjust as needed\nimport logging
import pandas as pd
import yaml
import json
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import numpy as np
# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load configuration
with open("../config/config.yaml", "r") as f:
    config = yaml.safe_load(f)
    
DATA_FILE = config["train_data"]
MODEL_FILE = config["model_file"]
SCORES_FILE = config["scores_file"]
TEST_SIZE = config["test_size"]
RANDOM_STATE = config["random_state"]
FEATURES = config["features"]
TARGET = config["target"]
MODEL_NAME = config["model"]["name"]
MODEL_PARAMS = config["model"]["params"]


def main():
    logging.info("Loading data...")
    data = pd.read_csv(DATA_FILE)

    X = data[FEATURES]
    y = data[TARGET]
    

    logging.info("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    logging.info("Training model...")
    if MODEL_NAME == "ridge":
        MODEL = Ridge(**MODEL_PARAMS)
    else:
        raise ValueError(f"{MODEL_NAME} is not supported.")
    
    pipeline = Pipeline([
        ("regression", MODEL)
    ])
    pipeline.fit(X_train, y_train)
    
    logging.info("Testing model...")
    y_pred = pipeline.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse) 
    mae = mean_absolute_error(y_test, y_pred)  
    r2 = r2_score(y_test, y_pred) 


    logging.info("Calculating scores...")
    logging.info(f"MSE: {mse:.4f}")
    logging.info(f"RMSE: {rmse:.4f}")
    logging.info(f"MAE: {mae:.4f}")
    logging.info(f"R-squared: {r2:.4f}")

    # Save scores to a JSON file
    scores = {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2
    }
    with open(SCORES_FILE, "w") as f:
        json.dump(scores, f, indent=4)

    logging.info("Saving model...")
    

    with open(MODEL_FILE, "wb") as file:
        pickle.dump(pipeline, file)
    logging.info("Model saved successfully.")
    logging.info(f"Model results saved to {SCORES_FILE}")

if __name__ == "__main__":
    main()
'''
    }
    
    for path, content in scripts.items():
        script_path = os.path.join(project_dir, path)
        with open(script_path, 'w') as f:
            f.write(content)
        print(f"Created script: {path}")

def run():
    project_name = "my_ml_project"
    # Create project directory
    project_dir = os.path.join(os.getcwd(), project_name)
    os.makedirs(project_dir, exist_ok=True)
    print(f"Created project directory: {project_dir}")
    
    # Create the folder structure and template files
    create_ml_project_structure(project_dir)
    
if __name__ == "__main__":
    run()
