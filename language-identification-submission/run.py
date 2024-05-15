from pathlib import Path

from joblib import load
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
from sklearn.metrics import f1_score

if __name__ == "__main__":

    # Load the data
    tira = Client()
    df = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", f"language-identification-validation-20240429-training"
    )



    # Load the model and make predictions
    model = load(Path(__file__).parent / "model.joblib")
    predictions = model.predict(df["text"])
    df["lang"] = predictions
    df = df[["id", "lang"]]
    print(len(df.index))

    # Save the predictions
    output_directory = get_output_directory(str(Path(__file__).parent))
    df.to_json(
        Path(output_directory) / "predictions.jsonl", orient="records", lines=True
    )


    # labels = tira.pd.truths(
    #    "nlpbuw-fsu-sose-24", f"language-identification-validation-20240429-training"
    # )

    # Calculate the F1 score
    # f1_score = f1_score(labels["lang"], predictions, average="weighted")
    # print(f"F1 score: {f1_score}")
