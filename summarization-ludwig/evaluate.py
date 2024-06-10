from rouge_score import rouge_scorer
from pathlib import Path

import pandas as pd
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory



if __name__ == "__main__":

    tira = Client()

    # Load predictions and targets
    output_directory = get_output_directory(str(Path(__file__).parent))
    prediction = pd.read_json(
        Path(output_directory) / "predictions.jsonl", orient="records", lines=True
    )


    targets_validation = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "summarization-validation-20240530-training"
    )

    # join the two dataframes on the id column
    # Convert the 'id' column in both DataFrames to string
    prediction['id'] = prediction['id'].astype(str)
    targets_validation['id'] = targets_validation['id'].astype(str)

    # Now you can merge
    merged_df = prediction.merge(targets_validation, on="id")
    # Export the merged dataframe as json with indentations for better readability
    merged_df.to_json("comparison.jsonl", orient="records",indent=4)

    # Export truths and prediction for each element as json
    targets_validation.to_json("targets_validation.jsonl", orient="records", lines=True)



    # Compute ROUGE scores
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}

    for target_summary, predicted_summary in zip(targets_validation["summary"], prediction["summary"]):
        scores = scorer.score(target_summary, predicted_summary)
        rouge_scores["rouge1"].append(scores["rouge1"].fmeasure)
        rouge_scores["rouge2"].append(scores["rouge2"].fmeasure)
        rouge_scores["rougeL"].append(scores["rougeL"].fmeasure)


    avg_rouge_scores = {metric: sum(scores) / len(scores) for metric, scores in rouge_scores.items()}

    print("ROUGE Scores:")
    print("-------------")
    print(f"ROUGE-1: {avg_rouge_scores['rouge1']}")
    print(f"ROUGE-2: {avg_rouge_scores['rouge2']}")
    print(f"ROUGE-L: {avg_rouge_scores['rougeL']}")
