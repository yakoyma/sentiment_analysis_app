"""
===============================================================================
This file contains all the functions for the project
===============================================================================
"""
# Libraries
import matplotlib.pyplot as plt
import pandas as pd


from wordcloud import WordCloud
from sklearn.metrics import (accuracy_score,
                             balanced_accuracy_score,
                             log_loss,
                             multilabel_confusion_matrix,
                             classification_report,
                             ConfusionMatrixDisplay)
from imblearn.metrics import classification_report_imbalanced



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


def evaluate_multiclass_classification(y_test, y_pred, y_proba, labels):
    """This function evaluates the result of a Multiclass Classification.

    Args:
        y_test (ndarray): the test labels
        y_pred (ndarray): the predicted labels
        y_proba (ndarray): the predicted probabilities
        labels (ndarray): list of unique labels for Confusion Matrix Plot
    """

    print('\nAccurcay: {:.3f}'.format(
        accuracy_score(y_true=y_test, y_pred=y_pred)))
    print('Balanced Accurcay: {:.3f}'.format(
        balanced_accuracy_score(y_true=y_test, y_pred=y_pred)))
    if y_proba is not None:
        print('Log loss: {:.3f}'.format(log_loss(
            y_true=y_test, y_pred=y_proba)))
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
