# Predicting Question Similarity

This project aims to implement the ideas in the paper [Siamese Recurrent Architectures for Learning Sentence Similarity](http://www.mit.edu/~jonasm/info/MuellerThyagarajan_AAAI16.pdf) on the [Quora Question pairs dataset](https://data.world/xprizeai-ai/quora-question-pairs).

You will need to download the dataset separately in order to run any of the code. If you wish to run the embedding model that uses pretrained GloVe embeddings, you will also need to download the [pretrained vectors](http://nlp.stanford.edu/data/glove.6B.zip).

## Summary

The dataset consists of pairs of questions labeled with a 0 if they are semantically different and 1 if they are semantically the same.

To embed the questions as vectors I implemented a model using pretrained GloVe embeddings, a model using Word2Vec embeddings trained on the dataset itself, and a model which uses a trainable embedding layer in the neural network itself.

Both the GloVe and Word2Vec models acheive an accuracy of around .815 on the test data, while the third model achieves an accuracy of about .84. However, this third model is significantly more prone to overfitting, and may not generalize as well to new data. 

A full accounting of the loss and accuracy for each epoch is provided in the histories folder.

## Running instructions

This code was written to be run in two steps. First you must decide which embedding model you wish to use. The options are to use pretrained global vectors (GloVe), a Word2Vec model which is trained on some subset of the data, or to simply train the word embeddings using an embedding layer in the neural network itself. You may choose which embedding dimension you would like to use, or use the default which is 300.

First run the file build_vocab.py along with an argument of either 'w2v', 'glove', or 'trainable'. You may assign a dimension using '--dimension dim' where dim is the dimension of your choice. Note that for the GloVe model the only options are 50, 100, 200, or 300. For example, running

```
python build_vocab.py glove
```

will assign each word in the dataset which is also in the GloVe corpus a unique integer, save that as a dictionary, then construct an embedding matrix using the pretrained vectors, and save that as a NumPy array file.

If you use the trainable model, there is no need to input a dimension until the next step, since no vectors are being created, it is simply building a dictionary between words and integers.

Once you have run build_vocab.py, the next step is to run process_and_train.py. Make sure you have downloaded the file model_architecure.py before running this file. Again you must specify either 'w2v', 'glove', or 'trainable', and you may specify an embedding dimension (default is 300), as well as how many epochs to train for (default is 10). An error will be thrown if you run the model with a dimension that you do not have vectors for. For example, running

```
python process_and_train.py glove --epochs 20
```

will convert the text data to sequences of integers, split into a training and testing set, and finally create and train the neural network for 20 epochs using a 300-dimensional embedding layer with pretrained weights. The best weights of the model will be saved so that you may evaluate it later. 

### Packages and GPU

This code was run using Python 3.6.8. and tensorflow 1.14 (nightly gpu). In particular, the model was run on GPU using the optimized for GPU CuDNNLSTM layer. Training time was approximately 90s per epoch on the GPU using the final configuration. The code can be run using the non-gpu version of tensorflow, although training time will be significantly slower. The code also requires pandas and numpy. 

If you plan to use the word2vec model, you will need to install the gensim package. The other packages used are part of the python standard library. 

## Outline

Prep Data:
1. Text cleaning
2. Tokenization (done automatically when using gensim's Word2Vec model)
    - Since we will be padding with zeros later, no word is assigned an integer of 0
3. Build embedding matrix (provided automatically when using Word2Vec)
    - The first row of the embedding matrix is initialized as a zero vector, to correspond to the 0 padding token
    - The final row of the embedding matrix is initialized as a random vector, to correspond to all out of vocabulary words
4. Replace text in dataset with sequences of integers
5. Split into training and testing sets (this is the same splitting in both models)
6. Pad sequences with zeros so they are all the same length

The neural network is built as follows:

Each question is input as a constant length sequence of integers into an embedding layer 
which takes each integer and represents it as a 300-dimensional vector. Each sequence of vectors is then fed into an LSTM
which outputs the final hidden layer (a 50-dimensional vector) that "encodes" the content of each sentence. These two 50-dimensional
vectors are then compared via a similarity function (in this case the manhattan distance) which is normalized so that two similar 
sentences will have similarity close to 1 and two dissimilar sentences will have similarity close to 0.

My model uses binary-crossentropy to measure the loss and is trained using the adam optimizer, which differs slightly from the
model outlined in the paper which uses Adagrad as its optimizer. In their case, the problem was not a binary classification problem
but rather assigned a similarity value between 1 and 5. However, as in the paper, we initialize the LSTM weights with small random
Gaussian entries and we increase the forget gate bias. 