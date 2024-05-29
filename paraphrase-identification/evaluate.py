from sklearn.metrics import matthews_corrcoef
from pathlib import Path


import pandas as pd
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory

if __name__ == "__main__":
    # load predictions and targets
    tira = Client()
    output_directory = get_output_directory(str(Path(__file__).parent))
    prediction = pd.read_json(
        Path(output_directory) / "predictions.jsonl", orient="records", lines=True
    )
    targets_validation = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "paraphrase-identification-validation-20240515-training"
    )
    mcc = matthews_corrcoef(targets_validation["label"], prediction["label"])
    print(f"MCC score: {mcc}")