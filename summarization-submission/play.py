from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
from pathlib import Path
from transformers import BertTokenizer, EncoderDecoderModel, Trainer, TrainingArguments, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import Dataset
import torch
import datasets


def preprocess_data(batch):
    # tokenize the inputs and labels
    inputs = tokenizer(batch["story"], padding="max_length", truncation=True, max_length=512)
    outputs = tokenizer(batch["summary"], padding="max_length", truncation=True, max_length=64)

    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask
    batch["decoder_input_ids"] = outputs.input_ids
    batch["decoder_attention_mask"] = outputs.attention_mask
    batch["labels"] = outputs.input_ids.copy()

    # We have to make sure that the PAD token is ignored for calculating the loss
    batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]]

    return batch



def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    # all unnecessary tokens are removed
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid

    return {
        "rouge2_precision": round(rouge_output.precision, 4),
        "rouge2_recall": round(rouge_output.recall, 4),
        "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
    }



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
    tokenizer.bos_token = tokenizer.cls_token
    tokenizer.eos_token = tokenizer.sep_token


    # Tokenize/ process the training dataset
    tokenized_dataset = training_dataset.map(preprocess_data, batched=True, remove_columns=["story", "summary"])

    # Convert the Dataset to PyTorch tensor with the expected columns
    tokenized_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids",
                           "decoder_attention_mask", "labels"])
       
    # roberta_shared = EncoderDecoderModel.from_encoder_decoder_pretrained("roberta-base", "roberta-base", tie_encoder_decoder=True)
    model = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-uncased', 'bert-base-uncased', tie_encoder_decoder=True)
    
    model.config.decoder_start_token_id = tokenizer.bos_token_id                                             
    model.config.eos_token_id = tokenizer.eos_token_id

    # model.config.pad_token_id = 
    model.config.max_length = 40
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4
    model.config.vocab_size = model.config.encoder.vocab_size
    # Create a data collator
    # data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Split the dataset into train and validation sets
    train_test_split = tokenized_dataset.train_test_split(test_size=0.1)
    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test']

    # load rouge for validation
    rouge = datasets.load_metric("rouge")

    # Set training arguments
    #        
    training_args = Seq2SeqTrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        predict_with_generate=True,
        eval_strategy="epoch",
        num_train_epochs=3,
        learning_rate=2e-5,
        do_train=True,
        do_eval=True,
        logging_steps=20, 
        save_steps=400, 
        eval_steps=500, 
        warmup_steps=500, 
        overwrite_output_dir=True,
        save_strategy="epoch",
        save_total_limit=3,
        weight_decay=0.01,
        metric_for_best_model="eval_rouge1",
        fp16=True
    )

    # Initialize the Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset, 
        tokenizer=tokenizer
    )

    # Train the model
    trainer.train()

    # Save the model
    model.save_pretrained("bert-summarization-model")
    tokenizer.save_pretrained("bert-summarization-tokenizer")

    # Inference
    model.eval()

