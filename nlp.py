import streamlit as st

#NLP pkgs
import spacy
import en_core_web_sm
from textblob import TextBlob
import gensim
from gensim.summarization import summarize
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
import nltk; nltk.download('punkt')

# Summary Fxn
def sumy_summarizer(docx):
    parser = PlaintextParser.from_string(docx, Tokenizer("english"))
    lex_summarizer = LexRankSummarizer()
    summary = lex_summarizer(parser.document, 3)
    summary_list = [str(sentence) for sentence in summary]
    result = ' '.join(summary_list)
    return result

def text_analyzer(my_text):
    nlp = spacy.load('en_core_web_sm')
    docx = nlp(my_text)

    tokens = [token.text for token in docx]
    alldata = [('"Token":{}, \n "Lemma":{}' .format(token.text, token.lemma_)) for token in docx]
    return alldata


def entity_analyzer(my_text):
    nlp = spacy.load('en_core_web_sm')
    docx = nlp(my_text)
    tokens = [token.text for token in docx]
    entities = [(entity.text, entity.label_) for entity in docx.ents]
    alldata = [f'Entity Text:{entities}, \n Token:{tokens}']
    return alldata

#Pkgs

def main():
    """ NLP App with Streamlit """
    st.title("NLP Apps with streamlit")

    # Tokenization
    if st.checkbox('Show tokens and Lemma'):
        st.subheader('Tokenize your text')
        message = st.text_area('Enter your text', key='token')
        if st.button('Analyze'):
            nlp_result = text_analyzer(message)
            st.json(nlp_result)

    # Named Entity
    if st.checkbox('Show named entities'):
        st.subheader('Extract entities from your text')
        message = st.text_area('Enter your text', key='entity')
        if st.button('Extract'):
            nlp_result = entity_analyzer(message)
            st.json(nlp_result)

    # Sentiment Analysis
    if st.checkbox('Show sentiment of input text'):
        st.subheader('Shows sentiment')
        message = st.text_area('Enter your text', key='sentiment')
        if st.button('Get Sentiment'):
            blob = TextBlob(message)
            result_sentiment = blob.sentiment
            st.success(result_sentiment)

    # Text Summarization
    if st.checkbox('Show text summarization'):
        st.subheader('Summarize your text')
        message = st.text_area('Enter your text', key='summarize')
        summary_options = st.selectbox("Select your summarizer",('gensim', 'sumy'))
        if st.button('Summarize'):
            if summary_options == 'gensim':
                st.text('Using Gensim...')
                summary_result = summarize(message)
            elif summary_options == 'sumy':
                st.text('Using Sumy..')
                summary_result = sumy_summarizer(message)
            else:
                st.warning('Using default summarizer..gensim')
                summary_result = summarize(message)

            st.success(summary_result)



if __name__ == '__main__':
    main()

