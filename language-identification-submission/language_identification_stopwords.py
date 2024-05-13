from pathlib import Path
import re

from tqdm import tqdm
import pandas as pd
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory

if __name__ == "__main__":

    tira = Client()

    # loading validation data (automatically replaced by test data when run on tira)
    #text_validation = tira.pd.inputs(
    #    "nlpbuw-fsu-sose-24", "language-identification-validation-20240429-training"
    #)

    # store text_validation as a JSON
    #text_validation.to_json("text_validation.json", orient="records", lines=True)
    # load text_validation from a JSON
    text_validation = pd.read_json("text_validation.json", orient="records", lines=True)

    #targets_validation = tira.pd.truths(
    #    "nlpbuw-fsu-sose-24", "language-identification-validation-20240429-training"
    #)

    # store targets_validation as a JSON
    #targets_validation.to_json("targets_validation.json", orient="records", lines=True)
    # load targets_validation from a JSON
    targets_validation = pd.read_json("targets_validation.json", orient="records", lines=True)

    lang_ids = [
        "af",
        "az",
        "bg",
        "cs",
        "da",
        "de",
        "el",
        "en",
        "es",
        "fi",
        "fr",
        "hr",
        "it",
        "ko",
        "nl",
        "no",
        "pl",
        "ru",
        "ur",
        "zh",
    ]

    stopwords = {
        lang_id: set(
            (Path(__file__).parent / "stopwords" / f"stopwords-{lang_id}.txt")
            .read_text()
            .splitlines()
        )
        - set(("(", ")", "*", "|", "+", "?"))  # remove regex special characters
        for lang_id in lang_ids
    }

    # classifying the data
    stopword_fractions = []
    for lang_id in tqdm(lang_ids):
        lang_stopwords = stopwords[lang_id]
        counts = pd.Series(0, index=text_validation.index, name=lang_id)
        for stopword in lang_stopwords:
            counts += (
                text_validation["text"]
                .str.contains(stopword, regex=False, case=False)
                .astype(int)
            )
        stopword_fractions.append(counts / len(lang_stopwords))
    stopword_fractions = pd.concat(stopword_fractions, axis=1)

    prediction = stopword_fractions.idxmax(axis=1)

    # converting the prediction to the required format
    prediction.name = "lang"
    prediction = prediction.to_frame()
    prediction["id"] = text_validation["id"]
    prediction = prediction[["id", "lang"]]

    # saving the prediction
    output_directory = get_output_directory(str(Path(__file__).parent))
    prediction.to_json(
        Path(output_directory) / "predictions.jsonl", orient="records", lines=True
    )

    # evaluate the prediction
    
