import re
import argparse
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
import numpy as np
import json
import os

def normalize_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9^/=+-]", " ", text)
    
    # Mathematical notation
    text = re.sub(r"/", " / ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"(\d)x(\d)","\g<1> x \g<2>",text)

    return text.split()

if __name__ == '__main__':
    data = pd.read_csv('quora_duplicate_questions.tsv', sep='\t')
    data.dropna(inplace=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="The embedding model you wish to use. Allowed values are glove, w2v, trainable", choices=['glove','w2v','trainable'])
    parser.add_argument("--dimension", help="embedding dimension", type=int)
    args = parser.parse_args()
    
    if args.dimension is not None:
        embedding_dim = args.dimension
    else:
        embedding_dim = 300

    if args.model == 'glove' and embedding_dim not in [50, 100, 200, 300]:
        parser.error("Permissible embedding dimensions for GloVe are 50, 100, 200, or 300")
    
    if not os.path.isdir(os.path.join('data',args.model)):
            os.makedirs(os.path.join('data',args.model))
            
    print('Using {} model with embedding dimension {}.'.format(args.model, embedding_dim), flush=True)
    
    if args.model == 'w2v':
        try:
            from gensim.models import Word2Vec
        except ImportError:
            "The gensim package must be installed in order to use this model"
        print('Compiling documents...', flush=True)
        questions = []
        for index, row in data.iterrows():
            questions.append(row['question1'])
            questions.append(row['question2'])
        unique_questions = list(set(questions))
        docs = [normalize_text(q) for q in unique_questions]
        print('Training w2v model...', flush=True)
        model = Word2Vec(docs, min_count = 5, size = embedding_dim)
        model.train(docs, total_examples=len(docs), epochs=10)
        print('Saving w2v model...', flush=True)
        filename = str(embedding_dim) + 'dim_w2v_model.bin'
        filepath = os.path.join('data','w2v',filename)
        model.save(filepath)
    
    else:
        print("Collecting vocabulary...", flush=True)
        word_data = []
        for index, row in data.iterrows():
            question1_words = normalize_text(row['question1'])
            question2_words = normalize_text(row['question2'])
            row_words = list(set(question1_words + question2_words))
            word_data += row_words
        unique_words = set(word_data) # keep as set for hashing
        vocab_size = len(unique_words)
        
        if args.model == 'glove':
            print("Creating embedding matrix...", flush=True)
            word_index = {}
            embedding_matrix = np.random.normal(scale=0.6, size=(vocab_size,embedding_dim))
            filepath = 'glove.6B/glove.6B.' + str(embedding_dim) + 'd.txt'
            
            with open(filepath, encoding='utf8') as f:
                for line in f:
                    word, *vector = line.split()
                    if word in unique_words:
                        idx = len(word_index) + 1
                        word_index[word] = idx
                        embedding_matrix[idx] = np.array(vector, dtype=np.float32)
            
            print('Saving pretrained glove model...', flush=True)
            filename = str(embedding_dim) + 'dim_glove_word_index.json'
            filepath = os.path.join('data','glove',filename)     
            with open(filepath,'w') as write_file:
                json.dump(word_index, write_file)

            vocab_size = len(word_index) + 1

            embedding_matrix = np.vstack((np.zeros((1, embedding_dim)), embedding_matrix[:vocab_size])) # first row for zero padding, final row for oov words
            
            filename = str(embedding_dim) + 'dim_glove_emb_matrix'
            filepath = os.path.join('data','glove',filename)
            
            np.save(filepath, embedding_matrix)
        
        if args.model == 'trainable':
            print("Tokenizing...", flush=True)
            t = Tokenizer(filters=[])
            t.fit_on_texts(unique_words)
            filepath = os.path.join('data', 'trainable', 'word_index.json')
            print("Saving word index...", flush=True)
            with open(filepath,'w') as write_file:
                json.dump(t.word_index, write_file)
    
    