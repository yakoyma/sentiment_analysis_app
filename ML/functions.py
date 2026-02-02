"""
===============================================================================
This file contains all the functions for the project
===============================================================================
"""
# Libraries
import numpy as np
import matplotlib.pyplot as plt


from wordcloud import WordCloud
from sklearn.metrics import (roc_auc_score,
                             log_loss,
                             accuracy_score,
                             balanced_accuracy_score,
                             multilabel_confusion_matrix,
                             classification_report,
                             ConfusionMatrixDisplay)



def display_pie_chart(dataset, var, figsize):
    """This function displays a pie chart with the proportions and
    count values.

    Args:
        dataset (pd.DataFrame): the Pandas dataset
        var (str): the variable (column of the dataset) to use
        title (str): the title of the chart
        figsize (tuple): the size of the chart
    """

    # Create a series with counted values
    dataviz = dataset[var].value_counts().sort_values(ascending=False)

    # Set up the figure
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('equal')
    ax.pie(
        x=list(dataviz),
        labels=list(dataviz.index),
        autopct='%1.1f%%',
        pctdistance=0.5,
        labeldistance=1.05,
        textprops=dict(color='black', size=12, weight='bold')
    )
    plt.title(f'{var} variable categories', size=18, weight='bold')
    plt.axis('equal')
    plt.grid(False)
    plt.show()


def display_barplot(dataset, var, figsize):
    """This function displays a barplot.

    Args:
        dataset (pd.DataFrame): the Pandas dataset
        var (str): the variable (column of the dataset) to use
        figsize (tuple): the size of the chart
    """

    # Create the dataset for visualisation
    dataviz = dataset[var].value_counts().sort_values(ascending=False)

    # Set up the figure
    ax = dataviz.plot.bar(figsize=figsize)
    for p in ax.patches:
        ax.annotate(
            str(p.get_height()),
            (p.get_x() + p.get_width() / 2, p.get_height()),
            ha='center',
            va='bottom'
        )
    ax.set_xlabel(var)
    ax.set_ylabel('Count')
    ax.set_title(f'Plot of variable {var}')
    ax.legend(loc='best')
    ax.grid(True)
    plt.show()


def display_wordcloud(dataset, var, color, SEED, figsize):
    """This function displays the Word Cloud of a text dataset.

    Args:
        dataset (pd.DataFrame): the Pandas dataset
        var (str): the text variable (column of the dataset) to use
        color (str): the color of the Word Cloud
        SEED (int): the random state value to use
        figsize (tuple): the size of the chart
    """

    corpus = []
    for i in range(len(dataset[var])):
        corpus.append(dataset[var].loc[i])
    text = ' '.join(corpus)

    # Instantiate the WordCloud
    wordcloud = WordCloud(
        stopwords=list(WordCloud().stopwords),
        random_state=SEED,
        background_color=color,
        normalize_plurals=True,
    )
    wordcloud.generate(text)
    plt.figure(figsize=figsize)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.grid(False)
    plt.show()


def load_datasets(X_train_path, X_test_path, y_train_path, y_test_path):
    """This function loads NumPy files of vector databases containing
    the training and test sets (features and labels).

    Args:
        X_train_path (str): the NumPy file path for train features
        X_test_path (str): the NumPy file path for test features
        y_train_path (str): the NumPy file path for train labels
        y_test_path (str): the NumPy file path for test labels

    Returns:
        X_train (array-like): the train set
        X_test (array-like): the test set
        y_train (array-like): the train target
        y_test (array-like): the test target
    """

    # Load NumPy files of vector databases containing the training and
    # test sets (features and labels)
    X_train = np.load(file=X_train_path)
    X_test = np.load(file=X_test_path)
    y_train = np.load(file=y_train_path)
    y_test = np.load(file=y_test_path)

    return X_train, X_test, y_train, y_test


def evaluate_multiclass_classification(y_test, y_pred, y_proba, labels):
    """This function evaluates the result of a Multiclass Classification.

    Args:
        y_test (array-like): the test labels
        y_pred (array-like): the predicted labels
        y_proba (array-like): the predicted probabilities
        labels (array-like): list of unique labels for Confusion Matrix Plot
    """

    if y_proba is not None:
        print('\n\nROC AUC: {:.3f}'.format(roc_auc_score(
            y_true=y_test, y_score=y_proba, multi_class='ovr')))
        print('Log loss: {:.3f}'.format(log_loss(
            y_true=y_test, y_pred=y_proba)))
    print('Accuracy: {:.3f}'.format(
        accuracy_score(y_true=y_test, y_pred=y_pred)))
    print('Balanced Accuracy: {:.3f}'.format(
        balanced_accuracy_score(y_true=y_test, y_pred=y_pred)))
    print('Multilabel Confusion Matrix:\n{}'.format(
        multilabel_confusion_matrix(y_true=y_test, y_pred=y_pred)))
    print('Classification Report:\n{}'.format(
        classification_report(y_true=y_test, y_pred=y_pred)))
    display = ConfusionMatrixDisplay.from_predictions(
        y_true=y_test,
        y_pred=y_pred,
        display_labels=labels,
        xticks_rotation='vertical',
        cmap=plt.cm.Blues
    )
    display.ax_.set_title('Plot of the Confusion Matrix')
    plt.grid(False)
    plt.show()
