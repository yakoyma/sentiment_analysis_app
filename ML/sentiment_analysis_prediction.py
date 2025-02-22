"""
===============================================================================
Multiclass Classification Project: Sentiment Analysis of Recipe Reviews and
User Feedback
===============================================================================
"""
# Standard libraries
import csv
import random
import platform
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Other libraries
import numpy as np
import pandas as pd
import sweetviz as sv
import ydata_profiling
import sklearn
import pickle
import pycaret


from sweetviz import analyze
from ydata_profiling import ProfileReport
from collections import Counter
from sklearn.model_selection import train_test_split
from pycaret.classification import *
from autogluon.tabular import TabularDataset, TabularPredictor
from functions import *


# Display versions of platforms and packages
print('\nPython: {}'.format(platform.python_version()))
print('NumPy: {}'.format(np.__version__))
print('Pandas: {}'.format(pd.__version__))
print('Sweetviz: {}'.format(sv.__version__))
print('YData-profiling: {}'.format(ydata_profiling.__version__))
print('Scikit-learn: {}'.format(sklearn.__version__))
print('PyCaret: {}'.format(pycaret.__version__))



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
# Load the dataset
print('\n\n\nLoad the dataset: ')
INPUT_CSV = 'dataset/Recipe Reviews and User Feedback Dataset.csv'
raw_dataset = pd.read_csv(INPUT_CSV)

# Display the raw dataset's dimensions
print('\nDimensions of the raw dataset: {}'.format(raw_dataset.shape))

# Display the raw dataset's information
print('\nInformation about the raw dataset:')
print(raw_dataset.info())

# Description of the raw dataset
print('\nDescription of the raw dataset:')
print(raw_dataset.describe(include='all'))

# Display the head and the tail of the raw dataset
print(f'\nRaw dataset shape: {raw_dataset.shape}')
print(raw_dataset.info())
print(pd.concat([raw_dataset.head(150), raw_dataset.tail(150)]))


# Dispaly the raw dataset report
report = analyze(source=raw_dataset)
report.show_html('raw_dataset_report.html')
#report_ydp = ProfileReport(df=raw_dataset, title='Raw Dataset Report')
#report_ydp.to_file('raw_dataset_report_ydp.html')


# Cleanse the dataset
dataset = raw_dataset[['user_id', 'text', 'stars']]
dataset = dataset.rename(columns={'stars': 'label'})
dataset['sentiment'] = dataset['label'].replace({
    0: 'Neutral',
    1: 'Very dissatisfied',
    2: 'Dissatisfied',
    3: 'Correct',
    4: 'Satisfied',
    5: 'Very satisfied'
})

# Management of rows
completion = dataset.count(axis='columns') / len(dataset.columns) * 100
completion = completion[completion < 100]
if completion.shape[0] > 0:
    dataset = dataset.drop(list(completion.index))

# Management of duplicates
print('\n\nManagement of duplicates:')
duplicate = dataset[dataset.duplicated()]
print('Dimensions of the duplicates dataset: {}'.format(duplicate.shape))
print(f'\nDuplicate dataset shape: {duplicate.shape}')
if duplicate.shape[0] > 0:
    dataset = dataset.drop_duplicates()

# Display the head and the tail of the duplicate
print(f'\nDuplicate shape: {duplicate.shape}')
print(duplicate.info())
print(pd.concat([duplicate.head(150), duplicate.tail(150)]))

# Management of missing data
dataset = dataset.dropna()
dataset.reset_index(inplace=True, drop=True)

# Display the head and the tail of the dataset
print(f'\nDataset shape: {dataset.shape}')
print(dataset.info())
print(pd.concat([dataset.head(150), dataset.tail(150)]))


# Dispaly the dataset report
report = analyze(source=raw_dataset)
report.show_html('dataset_report.html')
#report_ydp = ProfileReport(df=dataset, title='Dataset Report')
#report_ydp.to_file('dataset_report_ydp.html')


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
# Feature selection
y = dataset['label'].to_numpy()
X = dataset[['text', 'sentiment']]

# Display the head and the tail of the X dataset
print(f'\n\n\nX dataset shape: {X.shape}')
print(X.info())
print(pd.concat([X.head(150), X.tail(150)]))


# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED, shuffle=True)

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
# Classes and labels
print(f'\n\n\nTrain classes count: {Counter(y_train)}')
print(f'Test classes count: {Counter(y_test)}')
labels = list(set(X_test['sentiment']))
print(f'Labels: {labels}')


# 3.1 PyCaret
# Set up the setup
s = setup(
    data=train_dataset,
    target='label',
    index=False,
    test_data=test_dataset,
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
print(f'\n\nClassification of models:\n{best}')

# Make predictions
pred = predict_model(estimator=best)
print(pred)

# Evaluation
y_pred = pred['prediction_label'].to_numpy()
evaluate_multiclass_classification(y_test, y_pred, labels)

# Model persistence: save the pipeline
save_model(best, 'models/pycaret/model')

# Dashboard
#dashboard(best, display_format='inline')

# Create Gradio App
#create_app(best)


# 3.2 AutoGluon
train = TabularDataset(data=train_dataset)
test = TabularDataset(data=test_dataset)

# Instantiate AutoML instance
learner_kwargs = {'cache_data': False}
automl = TabularPredictor(
    label='label',
    problem_type='multiclass',
    eval_metric='accuracy',
    path='models/autogluon',
    log_file_path=False,
    learner_kwargs=learner_kwargs).fit(
    train_data=train,
    presets='best_quality'
)

# Display the best model
print('\n\nThe best model:\n{}'.format(
    automl.leaderboard(data=train, extra_info=True)))

# Make predictions
y_pred = np.array(automl.predict(test.drop(columns=['label']))).flatten()

# Summary
summary = automl.fit_summary()
print(summary)

# Evaluation
evaluate_multiclass_classification(y_test, y_pred, labels)
