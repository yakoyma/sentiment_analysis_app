"""
===============================================================================
Multiclass Classification Project: Sentiment Analysis of Recipe Reviews and
User Feedback using a Machine Learning (ML) model
===============================================================================

This file is organised as follows:
1. Data Analysis
2. Feature Engineering
3. Machine Learning
   3.1 PyCaret
   3.2 AutoGluon
   3.3 AutoKeras
"""
# Standard libraries
import random
import platform
import warnings

import keras.callbacks

# Suppress warnings
warnings.filterwarnings('ignore')

# Other libraries
import numpy as np
import pandas as pd
import sweetviz as sv
import ydata_profiling
import sklearn
import pycaret
import autokeras as ak


from sweetviz import analyze
from ydata_profiling import ProfileReport
from collections import Counter
from sklearn.model_selection import train_test_split
from pycaret.classification import *
from autogluon.tabular import TabularDataset, TabularPredictor
from autokeras import TextClassifier
from functions import *


# Display versions of platforms and packages
print('\n\nPython: {}'.format(platform.python_version()))
print('NumPy: {}'.format(np.__version__))
print('Pandas: {}'.format(pd.__version__))
print('Sweetviz: {}'.format(sv.__version__))
print('YData-profiling: {}'.format(ydata_profiling.__version__))
print('Scikit-learn: {}'.format(sklearn.__version__))
print('PyCaret: {}'.format(pycaret.__version__))
print('AutoKeras: {}'.format(ak.__version__))



# Constants
SEED = 0
MAX_ROWS_DISPLAY = 300
MAX_COLUMNS_DISPLAY = 150
FOLDS = 10

# Set the random seed for reproducibility
random.seed(SEED)
np.random.seed(SEED)

# Set the maximum number of rows and columns to display by Pandas
pd.set_option('display.max_rows', MAX_ROWS_DISPLAY)
pd.set_option('display.max_columns', MAX_COLUMNS_DISPLAY)



"""
===============================================================================
1. Data Analysis
===============================================================================
"""
print(f'\n\n\n1. Data Analysis')

# Load the dataset
INPUT_CSV = 'dataset/Recipe Reviews and User Feedback Dataset.csv'
raw_dataset = pd.read_csv(INPUT_CSV)

# Display the raw dataset's dimensions
print('\n\nDimensions of the raw dataset: {}'.format(raw_dataset.shape))

# Display the raw dataset's information
print('\nInformation about the raw dataset:')
print(raw_dataset.info())

# Description of the raw dataset
print('\nDescription of the raw dataset:')
print(raw_dataset.describe(include='all'))

# Display the head and the tail of the raw dataset
print(f'\nRaw dataset shape: {raw_dataset.shape}')
print(pd.concat([raw_dataset.head(150), raw_dataset.tail(150)]))


# Dispaly the raw dataset report
raw_dataset_report = analyze(source=raw_dataset)
raw_dataset_report.show_html('raw_dataset_report.html')
#report_ydp = ProfileReport(df=raw_dataset, title='Raw Dataset Report')
#report_ydp.to_file('raw_dataset_report_ydp.html')


# Cleanse the dataset
dataset = raw_dataset[['user_id', 'text', 'stars']]
dataset = dataset.rename(columns={'stars': 'label'})
dataset['sentiment'] = dataset['label'].replace(
    {
        0: 'Neutral',
        1: 'Very dissatisfied',
        2: 'Dissatisfied',
        3: 'Correct',
        4: 'Satisfied',
        5: 'Very satisfied'
    }
)

# Management of duplicates
print('\n\nManagement of duplicates:')
duplicate = dataset[dataset.duplicated()]
print('Dimensions of the duplicates dataset: {}'.format(duplicate.shape))
print(f'\nDuplicate dataset shape: {duplicate.shape}')
if duplicate.shape[0] > 0:
    dataset = dataset.drop_duplicates()
    dataset.reset_index(inplace=True, drop=True)

# Display the head and the tail of the duplicate
print(f'\nDuplicate shape: {duplicate.shape}')
print(duplicate.info())
print(pd.concat([duplicate.head(150), duplicate.tail(150)]))

# Management of missing data
if dataset.isna().any().any() == True:
    dataset = dataset.dropna()
    dataset.reset_index(inplace=True, drop=True)

# Display the dataset's dimensions
print('\nDimensions of the dataset: {}'.format(dataset.shape))

# Display the dataset's information
print('\nInformation about the dataset:')
print(dataset.info())

# Description of the dataset
print('\nDescription of the dataset:')
print(dataset.describe(include='all'))

# Display the head and the tail of the dataset
print(f'\nDataset shape: {dataset.shape}')
print(pd.concat([dataset.head(150), dataset.tail(150)]))


# Dispaly the dataset report
dataset_report = analyze(source=dataset)
dataset_report.show_html('dataset_report.html')
#dataset_report_ydp = ProfileReport(df=dataset, title='Dataset Report')
#dataset_report_ydp.to_file('dataset_report_ydp.html')


# Display the label categories
display_pie_chart(dataset, 'sentiment', (5, 5))
display_barplot(dataset, 'sentiment', (10, 5))


# Display the Word Cloud
display_wordcloud(dataset, 'text', 'white', SEED, (15, 6))



"""
===============================================================================
2. Feature Engineering
===============================================================================
"""
print(f'\n\n\n2. Feature Engineering')

# Feature selection
y = dataset['label'].to_numpy()
X = dataset[['text', 'sentiment']]

# Display the head and the tail of the dataset
print(f'\n\nX dataset shape: {X.shape}')
print(X.info())
print(pd.concat([X.head(150), X.tail(150)]))


# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y)

# Display the head and the tail of the train set
print(f'\n\nTrain set shape: {X_train.shape}')
print(X_train.info())
print(pd.concat([X_train.head(150), X_train.tail(150)]))

# Display the head and the tail of the test set
print(f'\nTest set shape: {X_test.shape}')
print(X_test.info())
print(pd.concat([X_test.head(150), X_test.tail(150)]))

# Display the training and test labels
print(f'\nTrain label shape: {y_train.shape}')
print(f'Test label shape: {y_test.shape}')


train_dataset = X_train.drop(['sentiment'], axis=1)
test_dataset = X_test.drop(['sentiment'], axis=1)
train_dataset = train_dataset.assign(label=y_train)
test_dataset = test_dataset.assign(label=y_test)

# Display the head and the tail of the train dataset
print(f'\n\nTrain dataset shape: {train_dataset.shape}')
print(train_dataset.info())
print(pd.concat([train_dataset.head(150), train_dataset.tail(150)]))

# Display the head and the tail of the test dataset
print(f'\nTest dataset shape: {test_dataset.shape}')
print(test_dataset.info())
print(pd.concat([test_dataset.head(150), test_dataset.tail(150)]))



"""
===============================================================================
3. Machine Learning
===============================================================================
"""
print(f'\n\n\n3. Machine Learning')

# Classes and labels
print(f'\n\nTrain classes count: {Counter(y_train)}')
print(f'Test classes count: {Counter(y_test)}')
labels = list(set(X_test['sentiment']))
print(f'Labels: {labels}')


# 3.1 PyCaret
print(f'\n\n3.1 PyCaret')

# Set up the setup
s = setup(
    data=train_dataset,
    target='label',
    index=False,
    train_size=0.8,
    text_features=['text'],
    preprocess=True,
    text_features_method='tf-idf',
    normalize=True,
    normalize_method='zscore',
    fold=FOLDS,
    fold_shuffle=True,
    n_jobs=-1,
    session_id=SEED,
    verbose=True
)

# Selection of the best model by cross-validation
best = compare_models(
    fold=FOLDS,
    round=3,
    cross_validation=True,
    n_select=1,
    turbo=True,
    sort='Accuracy',
    verbose=True
)
print(f'\nClassification of models:\n{best}')

# Make predictions
pred = predict_model(estimator=best, data=test_dataset)

# Evaluation
y_pred = pred['prediction_label'].values
y_proba=None
evaluate_multiclass_classification(y_test, y_pred, y_proba, labels)

# Model persistence: save the pipeline
save_model(best, 'models/pycaret/text/model')

# Dashboard
#dashboard(estimator=best)

# Create Gradio App
#create_app(estimator=best)


# 3.2 AutoGluon
print(f'\n\n3.2 AutoGluon')

train = TabularDataset(data=train_dataset)
test = TabularDataset(data=test_dataset)

# Instantiate AutoML instance
learner_kwargs = {'cache_data': False}
automl = TabularPredictor(
    label='label',
    problem_type='multiclass',
    eval_metric='accuracy',
    path='models/autogluon/text/model',
    log_file_path=False,
    learner_kwargs=learner_kwargs).fit(
    train_data=train,
    presets='best_quality'
)

# Display the best model
print('\nThe best model:\n{}'.format(
    automl.leaderboard(data=train, extra_info=True)))

# Make predictions
y_pred = np.array(automl.predict(test.drop(columns=['label']))).flatten()
if hasattr(automl, 'predict_proba'):
    y_proba = np.array(automl.predict_proba(test.drop(columns=['label'])))
else:
    y_proba=None

# Summary
summary = automl.fit_summary()
print(summary)

# Evaluation
evaluate_multiclass_classification(y_test, y_pred, y_proba, labels)


# 3.3 AutoKeras
print(f'\n\n3.3 AutoKeras')

# Instantiate the model
model = TextClassifier(
    multi_label=True,
    loss='categorical_crossentropy',
    metrics='accuracy',
    max_trials=1,
    objective='val_loss',
    overwrite=True,
    seed=SEED
)

# Train the model
model.fit(
    x=np.array(X_train[['text']]),
    y=y_train,
    epochs=5,
    validation_split=0.2,
    batch_size=32
)
print('\nSummary of the model with optimised hyperparameters:')
model = model.export_model()
print(model.summary())

# Make predictions
y_proba = model.predict(np.array(X_test[['text']]))

# Convert probabilities to classes
y_pred = np.argmax(y_proba, axis=1)

# Evaluation
evaluate_multiclass_classification(y_test, y_pred, y_proba, labels)

# Model persistence
model.save('models/autokeras/text/model')
