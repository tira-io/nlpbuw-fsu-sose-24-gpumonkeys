from tira.rest_api_client import Client
from joblib import dump
from sklearn.pipeline import Pipeline
from pathlib import Path
import spacy 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# MCC score: 0.214
class WordVectorSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, word_embedding_model):
        self.word_embedding_model = word_embedding_model

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        similarity_scores = []
        for index, row in X.iterrows():
            sentence1 = row["sentence1"]
            sentence2 = row["sentence2"]
            vector1 = self.word_embedding_model(sentence1).vector
            vector2 = self.word_embedding_model(sentence2).vector
            similarity_score = self.calculate_similarity(vector1, vector2)
            similarity_scores.append([similarity_score])
        return similarity_scores

    @staticmethod
    def calculate_similarity(vector1, vector2):
        # Calculate cosine similarity between two vectors
        similarity = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        return similarity



if __name__ == "__main__":

    # Load the data
    tira = Client()
    text = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "paraphrase-identification-train-20240515-training"
    ).set_index("id")
    labels = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "paraphrase-identification-train-20240515-training"
    ).set_index("id")
    
    df = text.join(labels)

    nlp = spacy.load("en_core_web_md")

    # train the model
    model = Pipeline(
        [
        ("word_vector_similarity", WordVectorSimilarity(word_embedding_model=nlp)),
        ("classifier", SVC())
    ]
    )
    # 
    model.fit(df[['sentence1', 'sentence2']], df['label'])

    # Save the model
    dump(model, Path(__file__).parent / "model.joblib")
