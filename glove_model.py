import pandas as pd
import numpy as np
import itertools
import re

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import ModelCheckpoint

from model_architecture import build_model

import json

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
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"(\d)x(\d)", "\g<1> x \g<2>", text)
    text = re.sub(r"\=", " = ", text)

    return text.split()
    
print('Building vocabulary...', flush=True)

word_data = []
for index, row in data.iterrows():
    question1_words = normalize_text(row['question1'])
    question2_words = normalize_text(row['question2'])
    row_words = list(set(question1_words + question2_words))
    word_data += row_words
    
t = Tokenizer(filters=[])
t.fit_on_texts(word_data)
word_to_int = t.word_index
#int_to_word = {v : k for k, v in word_to_int.items()}

vocab_size = len(word_to_int)

embedding_dim = 300 # options are 50, 100, 200, 300

filepath = 'glove.6B/glove.6B.' + str(embedding_dim) + 'd.txt'

print('Creating embedding matrix...', flush=True)

word_index = {}
embedding_matrix = np.random.normal(scale=0.6, size=(vocab_size,embedding_dim))
with open(filepath, encoding='utf8') as f:
    for line in f:
        word, *vector = line.split()
        if word in word_to_int:
            idx = len(word_index) + 1
            word_index[word] = idx
            embedding_matrix[idx] = np.array(vector, dtype=np.float32)
            
with open('data/glove_300/300dim_glove_word_index.json','w') as write_file:
    json.dump(word_index, write_file)

vocab_size = len(word_index) + 1

embedding_matrix = np.vstack((np.zeros((1, embedding_dim)), embedding_matrix[:vocab_size])) # zero padding

print('Vectorizing data...', flush=True)

def text_to_sequence(seq_of_words):
    text_as_ints = []
    for word in seq_of_words:
        index = word_index.get(word)
        if index == None:
            text_as_ints.append(0)
        else:
            text_as_ints.append(index)
    return text_as_ints
    
for index, row in data.iterrows():
    question1_words = normalize_text(row['question1'])
    question2_words = normalize_text(row['question2'])
    data.at[index,'question1'] = text_to_sequence(question1_words)
    data.at[index,'question2'] = text_to_sequence(question2_words)

X = data[['question1','question2']]
Y = data['is_duplicate']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, random_state=42)

X_train = {'left': X_train.question1, 'right': X_train.question2}
X_test = {'left': X_test.question1, 'right': X_test.question2}

# Make sure that both datasets are padded to the same length
max_seq_length = max(data.question1.map(lambda x: len(x)).max(), data.question2.map(lambda x: len(x)).max())

print('Padding sequences...', flush=True)

for dataset, side in itertools.product([X_train, X_test], ['left', 'right']): 
    dataset[side] = pad_sequences(dataset[side], maxlen=max_seq_length)

# Make sure that the data has the correct shape
assert X_train['left'].shape == X_train['right'].shape
assert X_test['left'].shape == X_test['right'].shape
assert len(X_train['left']) == len(Y_train)
assert len(X_test['left']) == len(Y_test)

np.save('data/glove_300/X_train_left.npy', X_train['left'])
np.save('data/glove_300/X_train_right.npy', X_train['right'])
np.save('data/glove_300/X_test_left.npy', X_test['left'])
np.save('data/glove_300/X_test_right.npy', X_test['right'])
np.save('data/glove_300/Y_train.npy', Y_train)
np.save('data/glove_300/Y_test.npy', Y_test)
np.save('data/glove_300/300dim_glove_emb_matrix', embedding_matrix)

model = build_model(vocab_size = vocab_size + 1, input_size = max_seq_length, embedding_dim = embedding_dim, lstm_units = 50, embedding_matrix = embedding_matrix, learning_rate = 0.001, clip_norm = None)

epochs = 50
batch_size = 64

checkpoint_path = "models\glove\glove_50ep_cp.ckpt"

cp_callback = ModelCheckpoint(checkpoint_path, monitor='val_acc', save_weights_only=True, save_best_only=True, mode='max', verbose=1)

model_trained = model.fit([X_train['left'], X_train['right']], Y_train, batch_size = batch_size, epochs = epochs, validation_split = 0.1, callbacks = [cp_callback], verbose=2)

history_file = 'histories/300dim_glove.txt'

score = model.evaluate([X_test['left'], X_test['right']], Y_test, verbose=0)
    
with open(history_file,'w') as file:
    file.write('300 dimensional GloVe embedding with 50 lstm units, no clip norm, and learning rate 0.001.\n')
    file.write(str(model_trained.history) + '\n')
    file.write('Test loss: {}\n'.format(score[0]))
    file.write('Test accuracy: {}'.format(score[1]))