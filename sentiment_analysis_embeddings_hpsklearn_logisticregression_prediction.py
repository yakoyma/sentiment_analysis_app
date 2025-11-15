"""
===============================================================================
Multiclass Classification Project: Sentiment Analysis of cooking Recipe Reviews
 and User Feedback
===============================================================================
"""
# Standard libraries
import random
import platform
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Other libraries
import numpy as np
import hyperopt


from hpsklearn import (HyperoptEstimator, logistic_regression)
from hyperopt import hp, tpe
from pickle import dump
from functions import *


# Display versions of platforms and packages
print('\n\nPython: {}'.format(platform.python_version()))
print('NumPy: {}'.format(np.__version__))
print('Hyperopt: {}'.format(hyperopt.__version__))



# Constant
SEED = 0

# Set the random seed for reproducibility
random.seed(SEED)
np.random.seed(SEED)



if __name__ == "__main__":

    # Load the training and test sets (features and labels)
    X_TRAIN_INPUT_NPY = 'dataset/embeddings/X_train.npy'
    X_TEST_INPUT_NPY = 'dataset/embeddings/X_test.npy'
    Y_TRAIN_INPUT_NPY = 'dataset/embeddings/y_train.npy'
    Y_TEST_INPUT_NPY = 'dataset/embeddings/y_test.npy'
    X_train, X_test, y_train, y_test = load_datasets(
        X_TRAIN_INPUT_NPY,
        X_TEST_INPUT_NPY,
        Y_TRAIN_INPUT_NPY,
        Y_TEST_INPUT_NPY
    )


    # Instantiate the HyperoptEstimator
    print('\n\nInstantiate the HyperoptEstimator')

    l1_ratio = hp.uniform('l1_ratio', 0, 1)
    estimator = HyperoptEstimator(
        classifier=logistic_regression('lr', l1_ratio=l1_ratio),
        algo=tpe.suggest,
        seed=SEED,
        verbose=True,
        n_jobs=-1
    )

    # Train the model
    estimator.fit(X_train, y_train)

    # Best hyperparameters of the model
    model = estimator.best_model()['learner']
    print('\nBest hyperparameters:\n{}'.format(model))

    # Make predictions
    y_pred = model.predict(X_test)
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)
    else:
        y_proba = None

    # Evaluation
    labels = list(set(y_test))
    evaluate_multiclass_classification(y_test, y_pred, y_proba, labels)

    # Model persistence
    dump(
        {'model': model},
        open('models/hpsklearn/embeddings/lr/model' + '.pkl', 'wb')
    )
