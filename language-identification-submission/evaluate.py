from sklearn.metrics import f1_score
from pathlib import Path
import re

from tqdm import tqdm
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
        "nlpbuw-fsu-sose-24", "language-identification-validation-20240429-training"
    )
    f1_score = f1_score(targets_validation["lang"], prediction["lang"], average="weighted")
    print(f"F1 score: {f1_score}")