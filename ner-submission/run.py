import nltk
from nltk import pos_tag, ne_chunk
import re
import pandas as pd
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
from pathlib import Path



# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('words', quiet=True)

def extract_entities(sentence):
    tokens = sentence.split()
    pos_tags = pos_tag(tokens)
    tree = ne_chunk(pos_tags)
    
    entities = []
    for subtree in tree:
        if isinstance(subtree, nltk.Tree):
            entity_type = subtree.label()
            entity_text = ' '.join([word for word, tag in subtree.leaves()])
            entities.append((entity_text, entity_type))
    
    return entities

def classify_time(token):
    time_patterns = [
        r'\d{1,2}:\d{2}',  # HH:MM
        r'\d{1,2}(am|pm)',  # HAM/PM
        r'\d{1,2}(:\d{2})?\s*(am|pm)',  # H:MM AM/PM or H AM/PM
        r'(january|february|march|april|may|june|july|august|september|october|november|december)',  # months
        r'(monday|tuesday|wednesday|thursday|friday|saturday|sunday)',  # days
        r'\d{1,2}/\d{1,2}/\d{2,4}',  # date patterns
        r'(today|tomorrow|yesterday)',
        r'\d{1,4}'  # years or days of month
    ]
    
    for pattern in time_patterns:
        if re.match(pattern, token.lower()):
            return True
    return False

def predict_ner(sentence):
    entities = extract_entities(sentence)
    tokens = sentence.split()
    tags = ['O'] * len(tokens)
    
    # First pass: identify all entities
    for entity_text, entity_type in entities:
        entity_tokens = entity_text.split()
        try:
            start_index = tokens.index(entity_tokens[0])
            if entity_type == 'PERSON':
                tag_prefix = 'PER'
            elif entity_type == 'ORGANIZATION':
                tag_prefix = 'ORG'
            elif entity_type == 'GPE':  # Geo-Political Entity
                tag_prefix = 'LOC'
            else:
                tag_prefix = entity_type
            
            tags[start_index] = f'B-{tag_prefix}'
            for i in range(start_index + 1, start_index + len(entity_tokens)):
                tags[i] = f'I-{tag_prefix}'
        except ValueError:
            continue  # If token not found, skip this entity
    
    # Second pass: identify time entities and fix organization tags
    time_sequence = False
    for i, token in enumerate(tokens):
        if tags[i] == 'O':
            if classify_time(token):
                if not time_sequence:
                    tags[i] = 'B-TIM'
                    time_sequence = True
                else:
                    tags[i] = 'I-TIM'
            else:
                time_sequence = False
        elif tags[i].endswith('ORG'):
            # Check if it's a known organization (you can expand this list)
            if token in ['Microsoft', 'OpenAI', 'Google', 'Apple', 'Amazon']:
                tags[i] = 'B-ORG'
        else:
            time_sequence = False
    
    # Third pass: ensure consistency in multi-token entities
    for i in range(1, len(tags)):
        if tags[i] == 'O':
            if tags[i-1].startswith('B-') or tags[i-1].startswith('I-'):
                prev_type = tags[i-1].split('-')[1]
                if tokens[i][0].isupper():  # Check if the current token starts with an uppercase letter
                    tags[i] = f'I-{prev_type}'
        elif tags[i-1] == 'B-PER' and tags[i] != 'I-PER' and tokens[i][0].isupper():
            tags[i] = 'I-PER'
    
    return tags


def process_dataframe(df):
    predictions = []
    for _, row in df.iterrows():
        sentence = row['sentence']
        predicted_tags = predict_ner(sentence)
        if len(predicted_tags) != len(sentence.split()):
            print(f"WARNING: Sentence '{sentence}' has {len(sentence.split())} tokens, but {len(predicted_tags)} tags were predicted.")
            print(f"Predicted tags: {predicted_tags}")
        predictions.append({'id': row['id'], 'tags': predicted_tags})
    return pd.DataFrame(predictions)


if __name__ == "__main__":
    tira = Client()
    # Load validation data (will be replaced by test data when run on TIRA)
    text = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "ner-validation-20240612-training"
    )

    # Process actual data
    prediction = process_dataframe(text)
    text = text.set_index("id")
    predictions = text.join(prediction.set_index("id"))
    df = predictions.drop(columns=["sentence"]).reset_index()
    
    # Save predictions
    output_directory = get_output_directory(str(Path(__file__).parent))
    df.to_json(
        Path(output_directory) / "predictions.jsonl",
        orient="records",
        lines=True
    )