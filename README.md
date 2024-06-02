# Dry Bean Classification Project

This project involves the classification of seven different types of dry beans using deep learning techniques with Keras. The dataset consists of various features extracted from high-resolution images of the beans.

## Project Structure

- `DryBeanDataset/`: Directory containing the dataset files.
  - `Dry_Bean_Dataset.arff`
  - `Dry_Bean_Dataset.txt`
  - `Dry_Bean_Dataset.xlsx`
- `main.py`: Main script for data preprocessing, model training, and evaluation.
- `venv/`: Virtual environment directory.

## Dataset

The dataset used for this project is the Dry Bean Dataset from the UCI Machine Learning Repository. It contains 13,611 instances of beans with 17 attributes each, including 16 features (such as area, perimeter, shape factors) and 1 target label (the class of the bean).

## Installation

1. **Clone the repository**:
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2. **Create a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Install additional dependencies**:
    ```bash
    pip install openpyxl tensorflow pandas numpy matplotlib seaborn scikit-learn
    ```

## Usage

To run the main script and train the model, use the following command:

```bash
python main.py
