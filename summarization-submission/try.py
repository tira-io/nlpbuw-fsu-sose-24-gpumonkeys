from transformers import BertTokenizer, EncoderDecoderModel
from tira.rest_api_client import Client
import pandas as pd
from tira.third_party_integrations import get_output_directory
from pathlib import Path



# Function to generate summaries
def generate_summary(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    summary_ids = model.generate(inputs.input_ids, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)



if __name__ == "__main__":
    tira = Client()
    df = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "summarization-validation-20240530-training"
    ).set_index("id")


    # Load the fine-tuned model and tokenizer
    model_path = ""
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = EncoderDecoderModel.from_pretrained(model_path)#

    # Generate summaries for the test set
    df['summary'] = df['story'].apply(generate_summary)

    df = df.drop(columns=["story"]).reset_index()

    # Save the predictions
    output_directory = get_output_directory(str(Path(__file__).parent))
    df.to_json(
        Path(output_directory) / "predictions.jsonl", orient="records", lines=True
    )