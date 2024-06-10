import datasets

# Load 1% of the training/validation sets.
train_data      = datasets.load_dataset("cnn_dailymail", "3.0.0", split="train[0:1%]")
validation_data = datasets.load_dataset("cnn_dailymail", "3.0.0", split="validation[0:1%]")

from transformers import BartTokenizer
from torch.utils.data import DataLoader

article_length = 512
summary_length = 64
batch_size     = 4

# Load the BART's pre-trained Tokenizer
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

# Define the function to make the correct data structure
def process_data_to_model_inputs(batch):
  # tokenize the inputs and labels
  inputs = tokenizer(batch["article"], padding="max_length", truncation=True, max_length=article_length)
  outputs = tokenizer(batch["highlights"], padding="max_length", truncation=True, max_length=summary_length)

  batch["input_ids"] = inputs.input_ids
  batch["attention_mask"] = inputs.attention_mask
  batch["decoder_input_ids"] = outputs.input_ids
  batch["decoder_attention_mask"] = outputs.attention_mask
  batch["labels"] = outputs.input_ids.copy()

  # We have to make sure that the PAD token is ignored for calculating the loss
  batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]]

  return batch

# Map the function to both train/validation sets.
train_data = train_data.map(
    process_data_to_model_inputs, 
    batched=True,
    remove_columns=["article", "highlights", "id"]
)

# Convert the Dataset to PyTorch tensor with the expected columns
train_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids",
                           "decoder_attention_mask", "labels"],
)

# Make the iterative object that does batching using the DataLoader
train_data      = DataLoader(train_data, batch_size=batch_size)

from transformers import BartForConditionalGeneration
import torch

# Load the model
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

# Put the model on GPU
if torch.cuda.is_available():
  model = model.to("cuda")

# Split model's components
the_encoder = model.get_encoder()
the_decoder = model.get_decoder()
last_linear_layer = model.lm_head

from torch.nn import CrossEntropyLoss
from transformers import AdamW
from transformers import get_scheduler

num_epochs = 3
num_training_steps = num_epochs * len(train_data)

# The loss function
loss_fct = CrossEntropyLoss()

# The optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

for epoch in range(num_epochs):

    model.train()
    for batch in train_data:
      if torch.cuda.is_available():
        batch = {k: v.to('cuda') for k, v in batch.items()}

      # Get the "input's representation"
      encoder_output = the_encoder(input_ids = batch['input_ids'],
                                   attention_mask = batch['attention_mask'])
      
      # Pass the representation + the target summary to the decoder
      decoder_output = the_decoder(input_ids=batch['decoder_input_ids'],
                                   attention_mask=batch['decoder_attention_mask'],
                                   encoder_hidden_states=encoder_output[0],
                                   encoder_attention_mask=batch['attention_mask'])

      # Use the last linear layer to predict the next token
      decoder_output = decoder_output.last_hidden_state
      lm_head_output = last_linear_layer(decoder_output)
      
      # Compute the loss
      loss = loss_fct(lm_head_output.view(-1, model.config.vocab_size),
                      batch['labels'].view(-1))

      loss.backward() # Update the weights
      optimizer.step() # Notify optimizer that a batch is done.
      lr_scheduler.step() # Notify the scheduler that a ...
      optimizer.zero_grad() # Reset the optimer

model.eval()
for batch in validation_data:
  if torch.cuda.is_available():
    batch = {k: v.to('cuda') for k, v in batch.items()}
        
  with torch.no_grad():
    outputs = model(**batch)

  loss = outputs.loss