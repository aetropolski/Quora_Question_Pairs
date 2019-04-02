import numpy as np
import pandas as pd
from build_vocab import normalize_text
from model_architecture import build_model
import os
import json
import argparse
import itertools

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def text_to_sequence(seq_of_words, word_to_int):
    """ Returns a sequence of words as a sequence of integers according to a provided dictionary. """
    vocab_size = len(word_to_int) + 1
    text_as_ints = []
    for word in seq_of_words:
        index = word_to_int.get(word)
        if index == None:
            text_as_ints.append(vocab_size)
        else:
            text_as_ints.append(index)
    return text_as_ints
    
if __name__ == '__main__':
    data = pd.read_csv('quora_duplicate_questions.tsv', sep='\t')
    data.dropna(inplace=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="The embedding model you wish to use. Allowed values are glove, w2v, trainable", choices=['glove','w2v','trainable'])
    parser.add_argument("--dimension", help="embedding dimension", type=int)    
    parser.add_argument("--epochs", help="number of training epochs", type=int)
    parser.add_argument("--history", help="save training history to text file", action="store_true")
    args = parser.parse_args()
    
    if args.dimension is not None:
        embedding_dim = args.dimension
    else:
        embedding_dim = 300
    
    if args.model == 'w2v':
        try:
            from gensim.models import Word2Vec
        except ImportError:
            "The gensim package must be installed in order to use this model"
        print('Loading word vectors...', flush=True)
        model_file = str(embedding_dim) + "dim_w2v_model.bin"
        model_path = os.path.join('data','w2v',model_file)
        assert os.path.isfile(model_path), "Run build_vocab.py with w2v argument and embedding dimension {} to initialize w2v model.".format(embedding_dim)
        model = Word2Vec.load(model_path)
        word_to_int = {w : model.wv.vocab.get(w).index + 1 for w in model.wv.vocab}
    if args.model == 'glove':
        print('Loading vocabulary...', flush=True)
        vocab_file = str(embedding_dim) + "dim_glove_word_index.json"
        vocab_path = os.path.join('data','glove',vocab_file)
        assert os.path.isfile(vocab_path), "Run build_vocab.py with glove argument and embedding dimension {} to initialize glove vocabulary.".format(embedding_dim)
        with open(vocab_path, 'r') as read_file:
            word_to_int = json.load(read_file)
    if args.model == 'trainable':
        print('Loading vocabulary...', flush=True)
        vocab_path = os.path.join('data','trainable','word_index.json')
        assert os.path.isfile(vocab_path), "Run build_vocab.py with trainable argument to initialize vocabulary for trainable embedding model."
        with open(vocab_path, 'r') as read_file:
            word_to_int = json.load(read_file)
    
    print('Encoding text...', flush=True)
    for index, row in data.iterrows():
        question1_words = normalize_text(row['question1'])
        question2_words = normalize_text(row['question2'])
        data.at[index,'question1'] = text_to_sequence(question1_words, word_to_int)
        data.at[index,'question2'] = text_to_sequence(question2_words, word_to_int)

    X = data[['question1','question2']]
    Y = data['is_duplicate']

    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, random_state=42)

    X_train = {'left': X_train.question1, 'right': X_train.question2}
    X_test = {'left': X_test.question1, 'right': X_test.question2}

    # Make sure that both datasets are padded to the same length
    max_seq_length = max(data['question1'].map(lambda x: len(x)).max(), data['question2'].map(lambda x: len(x)).max())

    print('Padding sequences...', flush=True)

    for dataset, side in itertools.product([X_train, X_test], ['left', 'right']): 
        dataset[side] = pad_sequences(dataset[side], maxlen=max_seq_length)
        
    vocab_size = len(word_to_int) + 1
    
    if args.model == 'w2v':
        embedding_matrix = np.vstack((np.zeros((1, embedding_dim)), model.wv.vectors, np.random.normal(scale=0.6, size=(1,embedding_dim))))
    if args.model == 'glove':
        emb_file = str(embedding_dim) + "dim_glove_emb_matrix.npy"
        emb_path = os.path.join('data','glove',emb_file)
        embedding_matrix = np.load(emb_path)
    if args.model == 'trainable':
        embedding_matrix = None
        vocab_size = vocab_size - 1 # this model has no oov words
        
    model = build_model(vocab_size = vocab_size + 1, input_size = max_seq_length, embedding_dim = embedding_dim, lstm_units = 50, embedding_matrix = embedding_matrix, learning_rate = 0.001, clip_norm = None)

    if args.epochs is not None:
        epochs = args.epochs
    else:
        epochs = 10
    batch_size = 64

    checkpoint_path = os.path.join("models", args.model, "cp.ckpt")

    cp_callback = ModelCheckpoint(checkpoint_path, monitor='val_acc', save_weights_only=True, save_best_only=True, mode='max', verbose=1)

    model_trained = model.fit([X_train['left'], X_train['right']], Y_train, batch_size = batch_size, epochs = epochs, validation_split = 0.1, callbacks = [cp_callback], verbose=2)
    
    if args.history:
        print('Saving history file...', flush = True)
        history_file = str(embedding_dim) + "dim_" + str(epochs) + "ep.txt"
        history_path = os.path.join('data',str(args.model), history_file)
        with open(history_path,'w') as file:
            file.write('{} dimensional {} model trained for {} epochs.\n'.format(embedding_dim, args.model, epochs))
            file.write(str(model_trained.history) + '\n')
