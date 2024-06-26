from pathlib import Path

from joblib import load
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory

if __name__ == "__main__":

    tira = Client()

    # loading validation data (automatically replaced by test data when run on tira)
    text_validation = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "ner-validation-20240612-training"
    )
    targets_validation = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "ner-validation-20240612-training"
    )

    model = load(Path(__file__).parent / "model.joblib")


    # labeling the data
    predictions = text_validation.copy()
    predictions['tags'] = predictions['sentence'].apply(lambda x: model.predict(x.split(" ")))
    predictions = predictions[['id', 'tags']]

    # saving the prediction
    output_directory = get_output_directory(str(Path(__file__).parent))
    predictions.to_json(
        Path(output_directory) / "predictions.jsonl", orient="records", lines=True
    )

    # evaluating the prediction
    from seqeval.metrics import f1_score

    y_true = [x['tags'] for x in targets_validation.to_dict(orient='records')]
    y_pred = [x['tags'] for x in predictions.to_dict(orient='records')]

    print(f1_score(y_true, y_pred))

