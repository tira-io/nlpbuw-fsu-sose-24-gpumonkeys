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
        # loading train data
    text_train = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "ner-training-20240612-training"
    ).set_index("id")
    targets_train = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "ner-training-20240612-training"
    ).set_index("id")
    # loading validation data (automatically replaced by test data when run on tira)
    text_validation = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "ner-validation-20240612-training"
    )
    targets_validation = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "ner-validation-20240612-training"
    )
    df = targets_train.join(text_train)

    from sys import exit  

    examples = []
    for i in range(len(df)):
        words = df.iloc[i]["sentence"].split(" ")
        labels = df.iloc[i]["tags"]
        sentence_examples = [{"word": words[j], "label": labels[j]} for j in range(len(words))]
        examples.append(sentence_examples)

    # Create a DataFrame
    import pandas as pd
    # convert the list of examples to a DataFrame
    df = pd.DataFrame([example for sentence in examples for example in sentence])

    

    # export the data to a json file with indent=4
    #df.to_json(Path(__file__).parent / "data.json", indent=4)

    # Train the model
    model = Pipeline(
        [
            ('vectorizer', CountVectorizer()),
        ('classifier', SVC(kernel='linear'))]
    )
    model.fit(df["word"], df["label"])

    # Save the model
    dump(model, Path(__file__).parent / "model.joblib")
