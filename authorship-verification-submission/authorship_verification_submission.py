from pathlib import Path

from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline




def evaluate_model(model, data, labels):
    predictions = model.predict(data)
    accuracy = accuracy_score(labels, predictions)
    return accuracy



if __name__ == "__main__":
    tira = Client()

    # loading train data
    text_train = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "authorship-verification-train-20240408-training"
    )
    targets_train = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "authorship-verification-train-20240408-training"
    )
    # loading validation data (automatically replaced by test data when run on tira)
    text_validation = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "authorship-verification-validation-20240408-training"
    )
    targets_validation = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "authorship-verification-validation-20240408-training"
    )

    tfidf_vectorizer = TfidfVectorizer(max_features=1000)

    # Model Training
    model = Pipeline([
        ('vectorizer', tfidf_vectorizer),
        ('classifier', LogisticRegression())
    ])

    model.fit(text_train['text'], targets_train['generated'])


    val_accuracy = evaluate_model(model, text_validation['text'], targets_validation['generated'])
    # print("Validation Accuracy:", val_accuracy)
    
    # make predictions
    predictions = model.predict(text_validation["text"])
    text_validation["generated"] = predictions
    df = text_validation[["id", "generated"]]

    # Save the predictions
    output_directory = get_output_directory(str(Path(__file__).parent))
    df.to_json(
        Path(output_directory) / "predictions.jsonl", orient="records", lines=True
    )