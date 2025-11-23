"""
===============================================================================
Multiclass Classification Project: Sentiment Analysis Application of Cooking
Recipe Reviews and User Feedback with a Machine Learning model and Gradio
===============================================================================
"""
# Standard library
import platform

# Other libraries
import numpy as np
import pandas as pd
import spacy
import demoji
import gradio as gr


from demoji import replace
from autogluon.tabular import TabularDataset, TabularPredictor


# Display versions of platforms and packages
print('\n\nPython: {}'.format(platform.python_version()))
print('NumPy: {}'.format(np.__version__))
print('Pandas: {}'.format(pd.__version__))
print('SpaCy: {}'.format(spacy.__version__))
print('Demoji: {}'.format(demoji.__version__))
print('Gradio: {}'.format(gr.__version__))



def get_prediction(max_tokens_length: int, text: str) -> str:
    """This function predicts the sentiment of a text using a trained
    Machine Learning (ML) model.

    Args:
        max_tokens_length (int): the maximum number of text tokens
        text (str): the user's text

    Returns:
        response (str): the predicted sentiment
    """

    try:

        # Check wether the user inputs are valid
        if max_tokens_length > 0 and text:

            # Instantiate the NLP model
            nlp = spacy.load(name='xx_ent_wiki_sm')
            nlp.add_pipe(factory_name='sentencizer')

            # Cleanse the text
            text = text.strip()
            text = replace(string=text, repl='')

            # Check if there is any text and wether the text tokens length at
            # the model input is less than the maximum tokens limit selected
            text_tokens_length = len(nlp(text))
            print(f'\n\nText tokens length: {text_tokens_length}')
            if (text_tokens_length > 0 and
                text_tokens_length < max_tokens_length):

                # Create dataset with the text data
                X = TabularDataset(data=pd.DataFrame(data={'text': [text]}))

                # Load the trained ML model
                model = TabularPredictor.load('models/autogluon/text/model')

                # Make prediction
                prediction = np.array(model.predict(X)).flatten()

                # Display the sentiment
                sentiments = {
                    'Neutral': 0,
                    'Very dissatisfied': 1,
                    'Dissatisfied': 2,
                    'Correct': 3,
                    'Satisfied': 4,
                    'Very satisfied': 5
                }
                sentiment = next(
                    key for key, value in sentiments.items() if
                    prediction[0] == value
                )
                response = (f'The predicted sentiment is: {sentiment}.')
            else:
                response = ('The text is too long and the maximum number of '
                            'tokens has been exceeded, or the text is '
                            'unreadable.')
        else:
            response = ('Invalid input data. Please complete the fields '
                        'correctly.')

    except Exception as error:
        response = f'The following unexpected error occurred: {error}'
    return response



# Instantiate the app
app = gr.Interface(
    fn=get_prediction,
    inputs=[
        gr.Slider(
            minimum=0,
            maximum=100000,
            value=1000,
            step=1000,
            label='Maximum text length'
        ),
        gr.Textbox(label='Text')
    ],
    outputs=gr.Textbox(label='Sentiment'),
    title='Recipe Reviews and User Feedback Sentiment Analysis Application'
)



if __name__ == '__main__':
    app.launch()
