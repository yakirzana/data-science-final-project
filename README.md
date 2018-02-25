# data-science-final-project
Final Project on data science.
## submitted by: Yakir Zana & Erez Bashari.

# project description:
collect watsapp chats and train modal according messages sent by men and women. generate a new message for men and women and classify thus messages using the trained model.

# STEP 1: Whatsapp parsing
[link to code](https://github.com/yakirzana/data-science-final-project/blob/master/whatsapp_parsing.ipynb)
###
we collect WhatsApp chats from our friend by asking them to send their chats history to us in a file format. then we remove msgs from senders (us), and ignore some chars(like emoticons). load all chats to array and save it for future steps.
##

# STEP 2: Build Classifier
[link to code](https://github.com/yakirzana/data-science-final-project/blob/master/classification.ipynb)
###
load men and women messages collected in step 1, select ~5000 randomly message each and start cleaning each message:
* remove non-hebrew letters (using regular expression)
* translate to English (to support steeming and stop words)
* steeming using PorterStemmer
* remove stop words
### create BOW
we adjust the number of words in the vocabulary to be 1000 because we see that give us the best predictions. save the vocabulary for step 4.
### train the model
split the all cleaned messages randomly to train & test.
### NB
simply train nb model and gain ~ 0.67 accuracy rate
### SVC
Initialize a SVC classifier with gamma = 0.001, C = 100, degree = 3 (default)
###
train SVC model and gain ~0.75 accuracy rate
## The Winner is SVC!
save it for step 4
##

# STEP 3: Word by word text generator with rnn using keras
[link to code](https://github.com/yakirzana/data-science-final-project/blob/master/kares_rnn.ipynb)
### This part divided into 3 steps, we go over all of them for women and men.
## Build The Model
Process the data for the model, do Vectorization, build the model with 2 stacked LSTM, and return the created model. If we have a pre-made model data on file, we load it.
###
We use softmax for the Activation function. The softmax function squashes the outputs of each unit to be between 0 and 1, just like a sigmoid function. But it also divides each output such that the total sum of the outputs is equal to 1 (check it on the figure above). The output of the softmax function is equivalent to a categorical probability distribution, it tells you the probability that any of the classes are true.
###
We use 2 stacked LSTM. LSTM is Long-Short Term Memory layer.
###
The original LSTM model is comprised of a single hidden LSTM layer followed by a standard feedforward output layer. The Stacked LSTM is an extension of this model that has multiple hidden LSTM layers where each layer contains multiple memory cells. Stacked LSTMs are now a stable technique for challenging sequence prediction problems. A Stacked LSTM architecture can be defined as an LSTM model comprised of multiple LSTM layers. An LSTM layer above provides a sequence output rather than a single value output to the LSTM layer below. Specifically, one output per input time step, rather than one output time step for all input time steps.
###
We use categorical_crossentropy as the losing function on the RNN.
###
We use rmsprop as the optimizer; rmspror divide the learning rate for a weight by a running average of the magnitudes of recent gradients for that weight.
###
## Train The Model
We use a checkpoint to save the weights to file because it takes a lot of time to make a full train. we update the file only if the loss a better score.
###
We use validation_split=0.05, for taking 5% of the data to be test data. We use batch_size=128, Batch size defines a number of samples that going to be propagated through the network. we found that 128 give us a good result, after some tests. with the limit of running time.
## Create Sentences
Take a seed from the data, and every time choose what will be the next word. 
###
save women and med generated messages for future steps.


# STEP 4: Classification of generated text sequences
[link to code](https://github.com/yakirzana/data-science-final-project/blob/master/classification_of_generated%20.ipynb)
###
In this step, we want to test our generator and predict message class using our trained model. 
###
first, we load the genreated women and men messages from step 3 and take amount like 30% of the initial data that we use to train the model.
###
clean the messages as we do in step 2 and create BOW using the vocabulary from step 2
###
load the classifier model from step 2 and use it to predict the gender of our generated messages.
## Results:
Men: 53.2% in prediction.
###
Women: 98.66666666666667% in prediction.
### Confusion Matrix
![](https://github.com/yakirzana/data-science-final-project/blob/master/cm.PNG)
### conclusion:
we can see that the classifier model work great on women class and less on the man class. this result can become because of many factors as we translated the Hebrew sentence to English and can be because that the man chats are from friends in the class and in the woman case it more wide topics.

