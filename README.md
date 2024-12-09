# NER-Mountains-Project

Project description:
For this project, the dataset was taken from the Kaggle platform: https://www.kaggle.com/datasets/geraygench/mountain-ner-dataset/data.

The notebook data_loading_and_preprocessing.ipynb performs the steps of loading, preprocessing 
and analysis of textual data containing markers for identifying mountain names. 
First, a dataset containing texts and markers indicating the positions of mountains in the text is loaded. 
For each line, the divide_markers function is applied, which divides the mountain names into separate words to more accurately determine
their positions in the text. Then two new columns are added: markers, which contains the split markers for each text, 
and mountain_count, which shows the number of mountains in each row. The get_mountains function extracts the names of the mountains
from the text based on the markers. After that, a graph is created to analyse the distribution of mountains in the dataset, 
showing how many mountains the texts contain and how often different mountain names occur. 
Finally, the frequency of each mountain in the dataset is calculated, which allows us to estimate which mountain names are most common 
in the dataset.


In the notebook data_preparation_for_model_evaluation.ipynb, the text is pre-processed and the model is evaluated
using the Precision, Recall and F1 metrics. First, the function process_text is defined, which cleans the text from punctuation, 
numbers and extra spaces, leaving only letters and converting the text to lowercase to ensure correct 
comparison. Next, the metric function is defined, which calculates the precision, recall, and average 
(F1) between the predicted and true results. For this purpose, text sets are transformed into 
sets, which allows us to correctly estimate the word match, and then calculate the metrics using the functions 
from the sklearn.metrics library. This notebook provides the necessary tools for pre-processing the data and evaluating it later, 
which is an important step in preparing a model for the Named Entity Recognition (NER) task.


The data_preparation.ipynb notebook prepares data for the Named Entity Recognition (NER) task. 
First, the texts are tokenised using SpaCy and the mountain names in the texts are labelled. 
Next, the processed data is saved to a CSV file. The dataset is then split into training and validation datasets, 
and K-Fold is used for cross-validation. We use vector representations of words (Word2Vec) 
vector representations of words (Word2Vec) are used to create text embeddings. The maximum length of tokens for training is also determined and a
a histogram of the distribution of the number of tokens in the texts. This stage of data preparation is important for further model training.

Modeling.py file: This file implements a model for extracting mountain names using the BERT transformational model, 
trained on the named entity recognition (NER) task. First, the tokeniser and the BERT model are loaded, 
configured to recognise entities of the ‘LOC’ type. Next, a pipeline for the NER is created, 
as well as a function for combining subtokens into full words.
The extract_mountain_names function uses this pipeline to process the texts and extracts the names of the mountains,
that are marked as ‘LOC’. After that, the prediction results for the training and validation (holdout) 
datasets are stored in separate columns. At the end of the program, predictions for several texts are displayed 
from the training set to check the results.