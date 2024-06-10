from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
from pathlib import Path
from transformers import BertTokenizer, EncoderDecoderModel, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset
import torch



# Preprocess the data
def preprocess_data(examples):
    inputs = examples['story']
    targets = examples['summary']
    # max_length=encoder_max_length
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=150, truncation=True, padding="max_length")
    model_inputs['labels'] = labels['input_ids']
    return model_inputs



# Function to generate summaries
def generate_summary(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    summary_ids = model.generate(inputs.input_ids, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)





if __name__ == "__main__":
    # Load the data
    tira = Client()
    df = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "summarization-validation-20240530-training"
    ).set_index("id")

    labels = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "summarization-validation-20240530-training"
    )

    # Join the datasets
    df = df.join(labels.set_index("id"))
    df = df.reset_index()
    df = df.drop(columns=["id"])
    # print(df.head())

    # Convert the dataframe to a Hugging Face Dataset
    training_dataset = Dataset.from_pandas(df)
    # print(training_dataset)

    # Load the tokenizer and model
    # tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # roberta_shared = EncoderDecoderModel.from_encoder_decoder_pretrained("roberta-base", "roberta-base", tie_encoder_decoder=True)
    model = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-uncased', 'bert-base-uncased')
    
    model.config.decoder_start_token_id = tokenizer.bos_token_id                                             
    model.config.eos_token_id = tokenizer.eos_token_id
    # Tokenize/ process the training dataset
    tokenized_dataset = training_dataset.map(preprocess_data, batched=True, remove_columns=["story", "summary"])

    # Create a data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Split the dataset into train and validation sets
    train_test_split = tokenized_dataset.train_test_split(test_size=0.1)
    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test']

    # Set training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=3
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    # Train the model
    trainer.train()

    # Save the model
    model.save_pretrained("bert-summarization-model")
    tokenizer.save_pretrained("bert-summarization-tokenizer")

    # Inference
    model.eval()


    # Generate summaries for the test set
    df['predicted_summary'] = df['story'].apply(generate_summary)

    # Prepare the predictions file
    df_predictions = df[['id', 'predicted_summary']].rename(columns={'predicted_summary': 'summary'})

    # Save the predictions
    output_directory = get_output_directory(str(Path(__file__).parent))
    df_predictions.to_json(
        Path(output_directory) / "predictions.jsonl", orient="records", lines=True
    )
