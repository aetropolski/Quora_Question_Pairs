# Predicting Question Similarity

This project aims to implement the ideas in the paper [Siamese Recurrent Architectures for Learning Sentence Similarity](http://www.mit.edu/~jonasm/info/MuellerThyagarajan_AAAI16.pdf) on the [Quora Question pairs dataset](https://data.world/xprizeai-ai/quora-question-pairs).

You will need to download the dataset separately in order to run any of the code. 

## Summary

The dataset consists of pairs of questions labeled with a 0 if they are semantically different and 1 if they are semantically the same.

To embed the questions as vectors I implemented a model using pretrained GloVe embeddings and a second model using Word2Vec embeddings trained on the dataset itself.

Both models acheive an accuracy of around .815 on the test data, although this could be improved slightly with early stopping.

A full accounting of the loss and accuracy for each epoch is provided in the histories folder.

### Packages and GPU

This code uses Python 3.6.8. A list of all installed packages and their versions is provided in the packages.txt file. 
In particular, the model was run on GPU using the optimized for GPU CuDNNLSTM layer. Training time was approximately 90s per epoch on the GPU using the final configuration.

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

### Files

model_architecture.py - Contains the build_model function which constructs and compiles the shared model.

glove_model.py - Processes all of the data using pretrained glove embeddings, then trains the model for 50 epochs. 
The history is stored in the histories folder. The weights of the model are also stored, as is the
processed data itself so that the model can be restored to compute predictions and further evaluate the model. 

word2vec_model.py - Processes all of the data using a word2vec model, then trains the model for 50 epochs. The history is stored
in the histories folder.