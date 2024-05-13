from pathlib import Path

from joblib import dump
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.pipeline import Pipeline
from tira.rest_api_client import Client

if __name__ == "__main__":

    # Load the data
    tira = Client()
    text = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "language-identification-validation-20240429-training"
    )
    text = text.set_index("id")
    labels = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "language-identification-validation-20240429-training"
    )
    df = text.join(labels.set_index("id"))

    # print the first 5 rows
    print(df.head())
    #print the number of rows
    print(len(df.index))
    

    # export the data to a json file with indent=4
    #df.to_json(Path(__file__).parent / "data.json", indent=4)

    # Train the model
    model = Pipeline(
        [        
        ('vectorizer', TfidfVectorizer()),
        ('classifier', SVC(kernel='linear'))]
    )
    model.fit(df["text"], df["lang"])

    # Save the model
    dump(model, Path(__file__).parent / "model.joblib")
