#!/usr/bin/env python
# coding: utf-8

# In[2]:


import re
import string

# Функція для попередньої обробки тексту
def process_text(text):
    # Видаляємо знаки пунктуації та цифри, залишаємо лише букви
    text = re.sub(r'[^\w\s]', '', text)  # Видалення пунктуації
    text = re.sub(r'\d', '', text)       # Видалення цифр
    text = re.sub(r'\s+', ' ', text)      # Видалення зайвих пробілів
    return text.lower()



# In[6]:


from sklearn.metrics import precision_score, recall_score, f1_score

# Функція для оцінки моделі за метриками Precision, Recall та F1
def metric(y_true, y_pred):
    # Перетворення наборів текстів у списки, які можна порівнювати
    y_true_processed = [set([process_text(s) for s in sample]) for sample in y_true]
    y_pred_processed = [set([process_text(s) for s in sample]) for sample in y_pred]

    # Генеруємо true positives, false positives та false negatives
    y_true_flat = []
    y_pred_flat = []

    for y_true_sample, y_pred_sample in zip(y_true_processed, y_pred_processed):
        y_true_flat.extend(list(y_true_sample))  # Перетворюємо множини у список
        y_pred_flat.extend(list(y_pred_sample))  # Перетворюємо множини у список

    # Розрахунок метрик
    precision = precision_score(y_true_flat, y_pred_flat, average='micro')
    recall = recall_score(y_true_flat, y_pred_flat, average='micro')
    f1 = f1_score(y_true_flat, y_pred_flat, average='micro')

    return {'precision': precision, 'recall': recall, 'f1': f1}


# In[ ]:




