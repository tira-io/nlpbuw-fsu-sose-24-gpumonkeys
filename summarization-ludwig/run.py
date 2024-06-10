from pathlib import Path
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from nltk.corpus import stopwords
from keybert import KeyBERT
import string
import nltk
import networkx as nx
import pandas as pd

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    
    # Convert to lowercase and remove punctuation
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize and remove stopwords
    words = nltk.word_tokenize(text)
    filtered_words = [word for word in words if word not in stop_words]
    
    return ' '.join(filtered_words)

def vectorize_sentences(sentences):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)
    return tfidf_matrix

def phrase_summarize(text, num_phrases=23):
    """    
    # Split the text into 3-grams
    sentences = nltk.sent_tokenize(text)
    
    if len(sentences) == 0:
        return ""
    
    # Preprocess sentences
    preprocessed_sentences = [preprocess_text(sentence) for sentence in sentences]
    
    # Use TfidfVectorizer to compute TF-IDF scores
    vectorizer = TfidfVectorizer(max_df=0.85, max_features=10000, ngram_range=(8, 8))
    tfidf_matrix = vectorizer.fit_transform(preprocessed_sentences)
    feature_names = vectorizer.get_feature_names_out()
    
    # Sum TF-IDF scores for each term
    tfidf_scores = tfidf_matrix.sum(axis=0).A1
    tfidf_scores_dict = dict(zip(feature_names, tfidf_scores))
    
    # Sort phrases by their TF-IDF scores
    sorted_phrases = sorted(tfidf_scores_dict.items(), key=lambda x: x[1], reverse=True)
    """
    # Initialize KeyBERT
    kw_model = KeyBERT()

    # Extract key phrases
    key_phrases = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=10)
    for phrase, score in key_phrases:
        print(f"Phrase: {phrase}, Score: {score}")

    
    # Select top key phrases
    key_phrases = [phrase for phrase, score in key_phrases[:num_phrases]]
    return " ".join(key_phrases)

if __name__ == "__main__":
    # Load the data
    tira = Client()
    df = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "summarization-validation-20240530-training"
    ).set_index("id")[:2]

    labels = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "summarization-validation-20240530-training"
    ).set_index("id")[:2]
    
    df['summary'] = df['story'].apply(lambda x: phrase_summarize(x, num_phrases=7))
    
    df = df.drop(columns=["story"]).reset_index()

    # Save the predictions
    output_directory = get_output_directory(str(Path(__file__).parent))
    df.to_json(
        Path(output_directory) / "predictions.jsonl", orient="records", lines=True
    )

# ROUGE Scores:
# -------------
# ROUGE-1: 0.3354224204495732
# ROUGE-2: 0.110518262170008
# ROUGE-L: 0.19676581970648405