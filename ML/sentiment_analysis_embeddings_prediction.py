"""
===============================================================================
Multiclass Classification Project: Sentiment Analysis of cooking Recipe Reviews
 and User Feedback
===============================================================================

This file is organised as follows:
1. Data Analysis
2. Feature Engineering
3. Machine Learning
   3.1 Perpetual ML
   3.2 FLAML
   3.3 AutoKeras
"""
# Standard libraries
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
import spacy
import sklearn
import flaml
import autokeras as ak
import keras


from sweetviz import analyze
from ydata_profiling import ProfileReport
from collections import Counter
from llama_index.embeddings.huggingface_openvino import OpenVINOEmbedding
from sklearn.model_selection import train_test_split, StratifiedKFold
from perpetual import PerpetualBooster
from flaml import AutoML
from autokeras import StructuredDataClassifier
from keras.callbacks import EarlyStopping
from pickle import dump, HIGHEST_PROTOCOL
from functions import *


# Display versions of platforms and packages
print('\n\nPython: {}'.format(platform.python_version()))
print('NumPy: {}'.format(np.__version__))
print('Pandas: {}'.format(pd.__version__))
print('Sweetviz: {}'.format(sv.__version__))
print('YData-profiling: {}'.format(ydata_profiling.__version__))
print('SpaCy: {}'.format(spacy.__version__))
print('Scikit-learn: {}'.format(sklearn.__version__))
print('FLAML: {}'.format(flaml.__version__))
print('AutoKeras: {}'.format(ak.__version__))
print('Keras: {}'.format(keras.__version__))



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

INPUT_CSV = 'dataset/Recipe Reviews and User Feedback Dataset.csv'
OUTPUT_EMBED = 'dataset/embeddings/embeddings.npy'



"""
===============================================================================
1. Data Analysis
===============================================================================
"""
print(f'\n\n\n1. Data Analysis')

# Load the dataset
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
y = dataset['label'].values
dataset = dataset[['text']]

# Display the head and the tail of the dataset
print(f'\n\nDataset shape: {dataset.shape}')
print(dataset.info())
print(pd.concat([dataset.head(150), dataset.tail(150)]))


# Create embeddings
corpus = dataset['text'].tolist()
print(f'\n\nCorpus length: {len(corpus)}')

# Instantiate the NLP model
nlp = spacy.load(name='xx_ent_wiki_sm')
nlp.add_pipe(factory_name='sentencizer')
corpus_text = '\n\n'.join(corpus)
corpus_tokens_length = len(nlp(corpus_text))
print(f'Corpus tokens length: {corpus_tokens_length}')

embedding_model = OpenVINOEmbedding(
    model_id_or_path='Snowflake/snowflake-arctic-embed-l-v2.0')
embeddings = embedding_model.get_text_embedding_batch(texts=corpus)
X = np.array(embeddings)
print(f'\n\nEmbeddings type: {type(X)}')
print(f'Embeddings length: {len(X)}')
print(f'Embeddings shape: {np.shape(X)}')

# Save embeddings into a NumPy file of vector databases
np.save(file=OUTPUT_EMBED, arr=X)

# Load the NumPy file of vector databases containing embeddings
print('\nLoad the NumPy file: ')
X = np.load(file=OUTPUT_EMBED)
print(f'Embeddings type: {type(X)}')
print(f'Embeddings length: {len(X)}')
print(f'Embeddings shape: {np.shape(X)}')


# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y)

# Display the training and test labels
print(f'\n\nTrain label shape: {y_train.shape}')
print(f'Test label shape: {y_test.shape}')

# Display the head and the tail of the train set
print(f'\nTrain shape: {X_train.shape}')
print(f'Test shape: {X_test.shape}')

# Save the training and test sets (features and labels) into NumPy files of
# vector databases
np.save(file='dataset/embeddings/X_train.npy', arr=X_train)
np.save(file='dataset/embeddings/X_test.npy', arr=X_test)
np.save(file='dataset/embeddings/y_train.npy', arr=y_train)
np.save(file='dataset/embeddings/y_test.npy', arr=y_test)



"""
===============================================================================
3. Machine Learning
===============================================================================
"""
print(f'\n\n\n3. Machine Learning')

# Classes and labels
print(f'\n\nTrain classes count: {Counter(y_train)}')
print(f'Test classes count: {Counter(y_test)}')
labels = list(set(y_test))
print(f'Labels: {labels}')


# 3.1 Perpetual
print(f'\n\n3.1 Perpetual ML')

# Instantiate the model
model = PerpetualBooster(objective='LogLoss')

# Train the model
model.fit(X_train, y_train)

# Best hyperparameters
print('\nBest hyperparameters:\n{}'.format(model.get_params()))

# Make predictions
y_pred = model.predict(X_test)
if hasattr(model, 'predict_proba'):
    y_proba = model.predict_proba(X_test)
else:
    y_proba=None

# Evaluation
evaluate_multiclass_classification(y_test, y_pred, y_proba, labels)

# Feature importance
print('\nTotal gain:\n{}'.format(
    model.calculate_feature_importance('TotalGain')))
print('Gain:\n{}'.format(model.calculate_feature_importance('Gain')))

# Model persistence
dump({'model': model}, open('models/perpetualbooster/model' + '.pkl', 'wb'))


# 3.2 FLAML
print(f'\n\n3.2 FLAML')

# Instantiate AutoML instance
automl = AutoML()
automl.fit(
    X_train=X_train,
    y_train=y_train,
    metric='log_loss',
    task='classification',
    n_jobs=-1,
    eval_method='auto',
    n_splits=FOLDS,
    split_type='auto',
    seed=SEED,
    early_stop=True
)

# Display information about the best model
print('\nBest estimator: {}'.format(automl.best_estimator))
print('Best hyperparameters:\n{}'.format(automl.best_config))
print('Best loss: {}'.format(automl.best_loss))
print('Training time: {}s'.format(automl.best_config_train_time))

# Make predictions
y_pred = automl.predict(X_test)
if hasattr(automl, 'predict_proba'):
    y_proba = automl.predict_proba(X_test)
else:
    y_proba=None

# Evaluation
evaluate_multiclass_classification(y_test, y_pred, y_proba, labels)

# Model persistence
model_path = 'models/flaml/model.pkl'
with open(model_path, 'wb') as f:
    dump(automl, f, HIGHEST_PROTOCOL)


# 3.3 AutoKeras
print(f'\n\n3.3 AutoKeras')

# Instantiate the model
model = StructuredDataClassifier(
    multi_label=True,
    loss='categorical_crossentropy',
    metrics='accuracy',
    max_trials=5,
    objective='val_loss',
    tuner='hyperband',
    overwrite=True,
    seed=SEED
)
early_stopping_callback = EarlyStopping(
    monitor='val_loss',
    verbose=0,
    mode='auto',
    patience=3
)

# Train the model
model.fit(
    x=X_train,
    y=y_train,
    epochs=500,
    validation_split=0.2,
    batch_size=32,
    callbacks=[early_stopping_callback]
)
print('\nSummary of the model with optimised hyperparameters:')
model = model.export_model()
print(model.summary())

# Make predictions
y_proba = model.predict(X_test)

# Convert probabilities to classes
y_pred = np.argmax(y_proba, axis=1)

# Evaluation
evaluate_multiclass_classification(y_test, y_pred, y_proba, labels)

# Model persistence
model_path = 'models/autokeras/model/model.keras'
model.save(model_path)
