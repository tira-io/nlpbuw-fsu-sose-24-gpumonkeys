from pathlib import Path
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from nltk.corpus import stopwords
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

def textrank_summarize(text, num_sentences=3):
    # Split the text into sentences
    sentences = nltk.sent_tokenize(text)
    
    if len(sentences) == 0:
        return ""
    
    # Preprocess sentences
    preprocessed_sentences = [preprocess_text(sentence) for sentence in sentences]
    
    # Vectorize the sentences
    tfidf_matrix = vectorize_sentences(preprocessed_sentences)
    
    # Compute the cosine similarity matrix
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    # Build the similarity graph
    nx_graph = nx.from_numpy_array(similarity_matrix)
    
    # Apply the PageRank algorithm
    scores = nx.pagerank(nx_graph)
    
    # Rank sentences based on scores
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    
    # Select the top num_sentences sentences
    summary_sentences = [sentence for score, sentence in ranked_sentences[:num_sentences]]
    
    # Combine selected sentences into the summary
    summary = ' '.join(summary_sentences)
    
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
    
    df['summary'] = df['story'].apply(lambda x: textrank_summarize(x, num_sentences=3))
    
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