import pandas as pd
import numpy as np
import itertools
import re

from gensim.models import Word2Vec
from gensim.test.utils import get_tmpfile
from gensim.models import KeyedVectors

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import ModelCheckpoint

from model_architecture import build_model

####################

data = pd.read_csv('quora_duplicate_questions.tsv', sep='\t')
data.dropna(inplace=True)

def normalize_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9^/=+-]", " ", text)
    
    # Common mathematical notation
    text = re.sub(r"/", " / ", text) # also separates words
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text) # also separates hyphenated words
    text = re.sub(r"(\d)x(\d)", "\g<1> x \g<2> ", text)
    text = re.sub(r"\=", " = ", text)

    return text.split()
    
documents = list(data['question1'].apply(normalize_text)) + list(data['question2'].apply(normalize_text))

embedding_dim = 300

print('Training embedding model...', flush=True)

model = Word2Vec(documents, min_count = 2, size = embedding_dim)
model.train(documents, total_examples=len(documents), epochs=10)

vocab_size = len(model.wv.vocab) + 1

embedding_matrix = np.vstack((np.zeros((1, embedding_dim)), model.wv.vectors, np.random.normal(scale=0.6, size=(1,embedding_dim))))

print('Vectorizing data...', flush=True)

def text_to_sequence(seq_of_words):
    text_as_ints = []
    for word in seq_of_words:
        index = model.wv.vocab.get(word)
        if index == None:
            text_as_ints.append(vocab_size)
        else:
            text_as_ints.append(index.index + 1)
    return text_as_ints
    
for index, row in data.iterrows():
    question1_words = normalize_text(row['question1'])
    question2_words = normalize_text(row['question2'])
    data.at[index,'question1'] = text_to_sequence(question1_words)
    data.at[index,'question2'] = text_to_sequence(question2_words)

X = data[['question1','question2']]
Y = data['is_duplicate']

# split into test/train
test_prop = 0.2

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=test_prop, random_state=42)

# Split into left/right datasets
X_train = {'left': X_train.question1, 'right': X_train.question2}
X_test = {'left': X_test.question1, 'right': X_test.question2}

# Make sure that both datasets are padded to the same length
max_seq_length = max(data.question1.map(lambda x: len(x)).max(), data.question2.map(lambda x: len(x)).max())

for dataset, side in itertools.product([X_train, X_test], ['left', 'right']): 
    dataset[side] = pad_sequences(dataset[side], maxlen=max_seq_length)

# Make sure that the data has the correct shape
assert X_train['left'].shape == X_train['right'].shape
assert X_test['left'].shape == X_test['right'].shape
assert len(X_train['left']) == len(Y_train)
assert len(X_test['left']) == len(Y_test)    

model = build_model(vocab_size = vocab_size + 1, input_size = max_seq_length, embedding_dim = embedding_dim, lstm_units = 50, embedding_matrix = embedding_matrix, learning_rate = 0.001, clip_norm = None)

epochs = 50
batch_size = 64

model_trained = model.fit([X_train['left'], X_train['right']], Y_train, batch_size = batch_size, epochs = epochs, validation_split = 0.1, verbose=2)

score = model.evaluate([X_test['left'], X_test['right']], Y_test, verbose=0)

history_file = 'histories/300dim_w2v_50epochs.txt'

print(score)
    
with open(history_file,'w') as file:
    file.write('300 dimensional w2v embedding with 50 lstm units, no clip norm, and learning rate 0.001.\n')
    file.write(str(model_trained.history) + '\n')
    file.write('Test loss: {}\n'.format(score[0]))
    file.write('Test accuracy: {}'.format(score[1]))