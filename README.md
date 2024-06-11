# OCWR (Object Classification with Radar)

Classify various human activities (6-class problem) using FMCW radar data.

## Introduction

Welcome to the OCWR project! This project aims to classify different human activities using Frequency-Modulated Continuous-Wave (FMCW) radar data. By processing the raw radar data and creating spectrograms, we can accurately identify various activities. We employed two different approaches to tackle this problem:
1. Traditional feature extraction and classification using KNN, Random Forest, XGBoost, and SVM classifiers.
2. Transfer learning approach by training only the first and last layer of the ResNet-18 and ResNet-34 models.

## Dataset

We used a dataset consisting of FMCW radar data collected during various human activities. The dataset, which is labeled with six different classes representing activities such as walking, sitting down, standing up, picking up an object, drinking water, and falling, can be found [here](https://researchdata.gla.ac.uk/848/).

## Installation

To get started with the OCWR project, follow these steps:
1. Clone the repository:
    ```bash
    git clone https://github.com/nikosfoto/ocwr.git
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Once you have cloned the repository and installed the dependencies, you can use the provided notebooks to generate spectrograms from the raw data, train the classification models, and evaluate their performance. Specifically:

- **Create spectrograms from raw data:** 
  Use `dataset_creation.ipynb` to generate spectrograms from the raw radar data. The numpy arrays will be saved in the sub-directory datasets/. The resulting NumPy arrays will be saved in the datasets/ sub-directory. To save time and avoid running the code to create spectrograms, you can download the pre-processed datasets/ folder [here](https://tud365-my.sharepoint.com/:f:/g/personal/gzanardini_tudelft_nl/ErwYcI17w9hAu0rksp3dFbgB7ZA5yfqqSi_R8DjhU7PYPw?e=aXuab9).
- **Traditional feature-based classifiers + evaluation:**
  Use `feature_based_classification.ipynb` to train and evaluate classifiers like KNN, Random Forest, XGBoost, and SVM.
- **ResNet-18 and ResNet-34 transfer learning:**
  Use `deepmodels_train.ipynb` to apply transfer learning on ResNet-18 and ResNet-34 models.
- **Evaluate the deep learning models:**
  Use `deepmodels_eval.ipynb` to assess the performance of the deep learning models.

## Contact
If you have any questions or suggestions, feel free to reach out to us at n.fotopoulos-1@student.tudelft.nl and g.zanardini@student.tudelft.nl
