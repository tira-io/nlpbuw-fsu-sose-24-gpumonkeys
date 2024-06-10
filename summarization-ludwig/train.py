from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
import torch
from transformers import BartTokenizer, BartForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_metric
from torch.utils.data import Dataset, DataLoader
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom Dataset Class
class SummarizationDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_input_length=1024, max_target_length=150):
        # Load the data
        tira = Client()
        df = tira.pd.inputs("nlpbuw-fsu-sose-24", "summarization-validation-20240530-training").set_index("id")
        labels = tira.pd.truths("nlpbuw-fsu-sose-24", "summarization-validation-20240530-training").set_index("id")
        
        # Merge the dataframes on the 'id' column
        self.data = df.merge(labels, on="id")
        self.texts = list(self.data['story'])
        self.summaries = list(self.data['summary'])
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.texts[idx]
        summary = self.summaries[idx]
        
        try:
            # Tokenize the inputs and labels
            model_inputs = self.tokenizer(
                text,
                max_length=self.max_input_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            labels = self.tokenizer(
                text_target=summary,
                max_length=self.max_target_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            # Ensure the labels tensor is correctly shaped for the model
            labels["input_ids"][labels["input_ids"] == self.tokenizer.pad_token_id] = -100

            # Flatten the tensors and return
            return {
                "input_ids": model_inputs["input_ids"].squeeze(),
                "attention_mask": model_inputs["attention_mask"].squeeze(),
                "labels": labels["input_ids"].squeeze()
            }
        except Exception as e:
            logger.error(f"Error processing index {idx}: {e}")
            raise e

# Initialize the tokenizer and model
model_checkpoint = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_checkpoint)
model = BartForConditionalGeneration.from_pretrained(model_checkpoint)

# Load dataset
try:
    dataset = SummarizationDataset(
        data_path='path_to_your_dataset.csv',  # Replace with your dataset path
        tokenizer=tokenizer
    )
except Exception as e:
    logger.error(f"Failed to load dataset: {e}")
    raise e

# Split dataset into train, validation, and test sets
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4)
test_loader = DataLoader(test_dataset, batch_size=4)

# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",  # Updated per the deprecation warning
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    logging_dir="./logs",
    logging_steps=10,
)

# Define compute_metrics function for evaluation
def compute_metrics(eval_pred):
    try:
        metric = load_metric("rouge")
        logits, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(logits, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Replace -100 in the labels as we can't decode them
        decoded_labels = [[label if label != -100 else tokenizer.pad_token_id for label in labels] for labels in labels]
        decoded_labels = tokenizer.batch_decode(decoded_labels, skip_special_tokens=True)
        
        # Compute ROUGE scores
        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        return result
    except Exception as e:
        logger.error(f"Error computing metrics: {e}")
        raise e

# Initialize the Seq2SeqTrainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=None,  # You can define a custom data collator if needed
    compute_metrics=compute_metrics
)

# Train the model
try:
    trainer.train()
except Exception as e:
    logger.error(f"Training failed: {e}")
    raise e

# Save the model
try:
    trainer.save_model("./trained_model")
except Exception as e:
    logger.error(f"Model saving failed: {e}")
    raise e

# Evaluate the model
try:
    results = trainer.evaluate(test_dataset)
    logger.info(f"Evaluation results: {results}")
except Exception as e:
    logger.error(f"Evaluation failed: {e}")
    raise e
