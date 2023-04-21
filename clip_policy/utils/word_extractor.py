import nltk 
from nltk.tokenize import word_tokenize 
from nltk.tag import pos_tag

def extract_from_query(query): # Extracts nouns from the sentence
    res = [] 

    query = nltk.word_tokenize(query)
    query = nltk.pos_tag(query) 

    for word, tag in query:
        if tag == 'NN':
            res.append(word)
    
    return res