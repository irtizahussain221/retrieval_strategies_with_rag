import nltk
from nltk.corpus import reuters
# from datasets import load_dataset
# import json


def get_articles(number_of_articles: int = 100):
    nltk.download("reuters")

    articles: list[str] = []
    i = 0
    for file_id in reuters.fileids():
        if i == number_of_articles:
            break

        articles.append(reuters.raw(file_id))
        i += 1

    return articles
    
# def get_articles(number_of_articles: int = 100):
#     articles = []
    
#     squad_dataset = load_dataset("squad")
#     train_data = squad_dataset['train']
    
#     i = 0
#     for entry in train_data:
#         if i == number_of_articles:
#             break
#         if entry['context'] not in articles:
#             i += 1
#             articles.append(entry['context'])
        
#     return articles