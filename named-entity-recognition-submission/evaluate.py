from pathlib import Path

from joblib import load
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
import pandas as pd

if __name__ == "__main__":

    tira = Client()

    targets_validation = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "ner-validation-20240612-training"
    )

    # load predictions
    output_directory = get_output_directory(str(Path(__file__).parent))
    predictions = pd.read_json(Path(output_directory) / "predictions2.jsonl", lines=True)

    # evaluating the prediction
    from seqeval.metrics import f1_score

    y_true = [x['tags'] for x in targets_validation.to_dict(orient='records')]
    y_pred = [x['tags'] for x in predictions.to_dict(orient='records')]

    print(f1_score(y_true, y_pred))

