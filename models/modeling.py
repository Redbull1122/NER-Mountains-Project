from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import pandas as pd
from tqdm import tqdm



# Завантаження токенізатора та NER-моделі
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
ner_model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")

# Створення пайплайна для NER
ner_pipeline = pipeline("ner", model=ner_model, tokenizer=tokenizer, grouped_entities=True)

# Функція для об'єднання субтокенів
def clean_tokens(tokens):
    """
    Об'єднує субтокени в одне слово.
    """
    return "".join(token[2:] if token.startswith("##") else f" {token}" for token in tokens).strip()

# Функція для витягування назв гір
def extract_mountain_names(texts, ner_pipeline):
    """
    Розпізнає назви гір у тексті за допомогою моделі NER.
    """
    mountain_names = []
    for text in tqdm(texts, desc="Processing NER"):
        entities = ner_pipeline(text)
        # Вибір сутностей з типом "LOC" (локальні назви, наприклад, гори)
        mountains = [
            clean_tokens(entity['word'].split()) for entity in entities
            if 'LOC' in entity['entity_group']
        ]
        mountain_names.append(" ".join(mountains))
    return mountain_names

# Завантаження обробленого датасету
processed_dataset = pd.read_csv('save_mountain_dataset_processed.csv')

# Розділення на train і holdout
train_dataset = processed_dataset.iloc[:1400].reset_index(drop=True)
holdout_dataset = processed_dataset.iloc[1400:].reset_index(drop=True)

# Формування текстів із токенів
train_texts = train_dataset['tokens'].apply(lambda x: " ".join(x.split(',')) if isinstance(x, str) else "").tolist()
holdout_texts = holdout_dataset['tokens'].apply(lambda x: " ".join(x.split(',')) if isinstance(x, str) else "").tolist()

# Виконання інференсу для тренувального датасету
train_dataset['predicted_mountains'] = extract_mountain_names(train_texts, ner_pipeline)

# Виконання інференсу для holdout-датасету
holdout_dataset['predicted_mountains'] = extract_mountain_names(holdout_texts, ner_pipeline)

# Перевірка результатів
print("Прогноз для 50 текстів:")
for i, text in enumerate(train_dataset['predicted_mountains'].iloc[50:100], start=1):
    print(f"Текст {i}: {text}")



