"""
===============================================================================
Multiclass Classification Project: Sentiment Analysis Application with an
Artificial Intelligence (AI) model and Gradio
===============================================================================
"""
# Standard library
import platform

# Other libraries
import langchain_community
import langchain_core
import spacy
import gradio as gr


from langchain_community.document_loaders import AsyncHtmlLoader, PyPDFLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from gradio_pdf import PDF


# Display versions of platforms and packages
print('\n\nPython: {}'.format(platform.python_version()))
print('LangChain Community: {}'.format(langchain_community.__version__))
print('LangChain Core: {}'.format(langchain_core.__version__))
print('SpaCy: {}'.format(spacy.__version__))
print('Gradio: {}'.format(gr.__version__))



def get_response(language: str, url: str, file: str, text: str) -> str:
    """This function predicts the sentiment of a text using an AI model.

    Args:
        language (str): the language of the text
        url (str): the website url to scrape
        file (str): the path of the PDF file
        text (str): the user's text

    Returns:
        response (str): the response of the model with the text's sentiment
    """
    try:

        # Check wether the user inputs are valid
        if language and any([url, file, text]):

            # Load the dataset
            if url:
                # Web scraping
                url = url.strip()
                loader = AsyncHtmlLoader(
                    web_path=[url], default_parser='html.parser')
                html_document = loader.load()
                document = BeautifulSoupTransformer().transform_documents(
                    html_document)
                text = ''.join(doc.page_content for doc in document)
            elif file: # Check if there is any PDF file
                # Load the PDF file
                loader = PyPDFLoader(file_path=file)
                document = []
                for page in loader.lazy_load():
                    document.append(page)
                text = ''.join(doc.page_content for doc in document)

            # Instantiate the NLP model
            nlp = spacy.load(name='xx_ent_wiki_sm')
            nlp.add_pipe(factory_name='sentencizer')

            # Cleanse the text
            text = text.strip()

            # Create the prompt template
            template = """
            You are a nice and helpful assistant, and your task is to provide
            the Sentiment Analysis of the following text:

            Text: {text}

            by choosing one of the following words for your answer: Very 
            Negative, Negative, Neutral, Correct, Positive, and Very Positive. 
            The language of your answer must be {language}, the same as that 
            of the text."""

            # Check if there is any text and wether the text tokens length at
            # the model input is less than the maximum tokens limit selected
            text_tokens_length = len(nlp(text))
            template_tokens_length = len(nlp(template))
            input_tokens_length = (text_tokens_length + template_tokens_length)
            if text_tokens_length > 0 and input_tokens_length < 2500:

                # Select the context window size
                num_ctx = next(pow(2, i) for i in range(7, 13) if
                    input_tokens_length < pow(2, i))

                # Instantiate the model
                model = Ollama(
                    model='alibayram/erurollm-9b-instruct',
                    num_ctx=num_ctx,
                    num_predict=256,
                    temperature=0.7,
                    top_k=40,
                    top_p=0.8
                )
                prompt = PromptTemplate.from_template(template)
                chain = prompt | model | StrOutputParser()
                response = chain.invoke({'text': text, 'language': language})
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
languages_list = [
    'Arabic', 'Bulgarian', 'Catalan', 'Chinese', 'Croatian', 'Czech', 'Danish',
    'Dutch', 'English', 'Estonian', 'Finnish', 'French', 'Galician', 'German',
    'Greek', 'Hindi', 'Hungarian', 'Irish', 'Italian', 'Japanese', 'Korean',
    'Latvian', 'Lithuanian', 'Maltese', 'Norwegian', 'Polish', 'Portuguese',
    'Romanian', 'Russian', 'Slovak', 'Slovenian', 'Spanish', 'Swedish',
    'Turkish', 'Ukrainian'
]
app = gr.Interface(
    fn=get_response,
    inputs=[
        gr.Dropdown(
            choices=languages_list,
            label='Source and Answer languages (Supported languages)',
            type='value'
        ),
        gr.Textbox(label='url'),
        PDF(label='PDF file'),
        gr.Textbox(label='Text in supported languages')
    ],
    outputs=gr.Textbox(label='Response'),
    title='Sentiment Analysis Application'
)



if __name__ == '__main__':
    app.launch()
