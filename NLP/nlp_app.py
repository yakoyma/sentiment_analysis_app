"""
===============================================================================
Multiclass Classification Project: Sentiment Analysis Application with a
Natural Language Processing (NLP) Classifier and Gradio
===============================================================================
"""
# Standard library
import platform

# Other libraries
import demoji
import transformers
import gradio as gr


from demoji import replace
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          pipeline)
from gradio_pdf import PDF


# Display versions of platforms and packages
print('\n\nPython: {}'.format(platform.python_version()))
print('Demoji: {}'.format(demoji.__version__))
print('Transformers: {}'.format(transformers.__version__))
print('Gradio: {}'.format(gr.__version__))



def get_prediction(language: str, text: str) -> str:
    """This function predicts the sentiment of a text using a NLP
    Classification model.

    Args:
        language (str): the language of the text
        text (str): the user's text

    Returns:
        response (str): the predicted sentiment and the score of prediction
    """

    try:

        # Check wether the user inputs are valid
        if language and text:

            # Instantiate the classification model
            model_name = 'tabularisai/multilingual-sentiment-analysis'
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            pipe = pipeline(
                task='text-classification', model=model, tokenizer=tokenizer)

            # Cleanse the text
            text = text.strip()
            text = replace(string=text, repl='')

            # Check if there is any text and wether the text tokens length at
            # the model input is less than the maximum tokens limit selected
            text_tokens_length = len(tokenizer.tokenize(text))
            if text_tokens_length > 0 and text_tokens_length < 512:

                # Get the sentiment and the score of prediction
                sentiment = pipe(text)[0]['label']
                score = round(pipe(text)[0]['score'], 2)
                response = (f'The predicted sentiment of the text in '
                            f'{language} is: {sentiment}.'
                            f'\nThe score of prediction is: {score}.')
            else:
                response = ('The text is too long and the maximum number of '
                            'tokens has been exceeded, or the text is '
                            'unreadable.')
        else:
            response = ('Invalid input data. Please complete the field '
                        'correctly.')

    except Exception as error:
        response = f'The following unexpected error occurred: {error}'
    return response



# Instantiate the app
languages_list = [
    'Arabic (العربية)', 'Bengali (বাংলা)', 'Chinese (中文)',
    'French (Français)', 'German (Deutsch)', 'Hindi (हिन्दी)',
    'Italian (Italiano)', 'Japanese (日本語)', 'Korean (한국어)',
    'Malay (Bahasa Melayu)', 'Polish (Polski)', 'Portuguese (Português)',
    'Russian (Русский)', 'Spanish (Español)',
    'Swiss German (Schweizerdeutsch)', 'Tagalog', 'Dutch (Nederlands)',
    'Telugu (తెలుగు)', 'Turkish (Türkçe)', 'Ukrainian (Українська)',
    'Vietnamese (Tiếng Việt)'
]
app = gr.Interface(
    fn=get_prediction,
    inputs=[
        gr.Dropdown(
            choices=languages_list,
            label='Language of the text (Supported languages)',
            type='value'
        ),
        gr.Textbox(label='Text in supported languages')
    ],
    outputs=gr.Textbox(label='Sentiment'),
    title='Sentiment Analysis Application'
)



if __name__ == '__main__':
    app.launch()
