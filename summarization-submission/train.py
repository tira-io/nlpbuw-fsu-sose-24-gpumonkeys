from pathlib import Path
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from nltk.corpus import stopwords
import string
import nltk
import pandas as pd


def preprocess_text(sentences):
    # Convert sentences to lowercase, remove punctuation, remove stopwords, and tokenize
    stop_words = set(stopwords.words('english'))
    processed_sentences = []
    
    for sentence in sentences:
        # Convert to lowercase
        sentence = sentence.lower()
        
        # Remove punctuation
        sentence = sentence.translate(str.maketrans('', '', string.punctuation))
        
        # Remove stopwords
        words = sentence.split()
        filtered_text = [word for word in words if word not in stop_words]
        
        # Tokenize
        tokens = nltk.word_tokenize(' '.join(filtered_text))
        
        if tokens:  # Only add non-empty sentences
            processed_sentences.append(' '.join(tokens))
    
    return processed_sentences

# def preprocess_text(text):
#         # Convert text to lowercase
#         text = text.lower()
        
#         # Remove punctuation
#         text = text.translate(str.maketrans('', '', string.punctuation))
        
#         # Remove stopwords
#         stop_words = set(stopwords.words('english'))
#         words = text.split()
#         filtered_text = [word for word in words if word not in stop_words]
        
#         # Tokenize text
#         tokens = nltk.word_tokenize(' '.join(filtered_text))
        
#         return ' '.join(tokens)

def vectorize_sentences(sentences):
    if not sentences:
        return None
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)
    return tfidf_matrix


def rank_sentences(tfidf_matrix):
    if tfidf_matrix is None:
        return []
    
    # Compute pairwise cosine similarity between sentences
    similarity_matrix = cosine_similarity(tfidf_matrix)
    # Rank sentences based on similarity scores
    ranked_sentences = np.argsort(np.sum(similarity_matrix, axis=1))[::-1]
    return ranked_sentences


def summarize(text):
    # Split the text into sentences
    sentences = nltk.sent_tokenize(text)

    # Preprocess each sentence
    pp_sentences = preprocess_text(sentences=sentences)
    
    # Vectorize each sentence
    tfidf_matrix = vectorize_sentences(sentences)
    
    # Calculate similarity matrix for each sentence
    ranked_sentences = rank_sentences(tfidf_matrix)
    
    top_n_sentences = [sentences[i] for i in ranked_sentences[:3]]
    summary = ' '.join(top_n_sentences)
    
    return summary


if __name__ == "__main__":

    # Load the data
    tira = Client()
    df = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "summarization-validation-20240530-training"
    ).set_index("id")

    labels = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "summarization-validation-20240530-training"
    ).set_index("id")
    


    # Extractive summarization

    df['summary'] = df['story'].apply(summarize)
    
    df = df.drop(columns=["story"]).reset_index()

    # Save the predictions
    output_directory = get_output_directory(str(Path(__file__).parent))
    df.to_json(
        Path(output_directory) / "predictions.jsonl", orient="records", lines=True
    )

# ROUGE Scores:
# -------------
# ROUGE-1: 0.32716691567928685
# ROUGE-2: 0.1103239048304118
# ROUGE-L: 0.19552394930314196