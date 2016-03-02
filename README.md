# SpamHam
classifies a given email as spam or ham(non-spam)

1. Training directory has 10 sub-directories containing spam and non-spam mails, along with the subject.
The spam mails have file names starting with 'spm'. The data set is not balanced. Spam mails
are much fewer than non-spam ones. Use the naive Bayes algorithm to build a classier to classify
mail as spam or non-spam. 10-fold cross validated is used for training and validation of the model.
(a) on the mail subject+body as is.
(b) remove stop words.
(c) remove stop words and lemmatize the email.

2. spam/non-spam classifier using a linear discriminant function/perceptron method using a bag of words
(BoW) representation in three different ways:
(a) Use a binary BoW representation.
(b) Use a term frequency based BoW representation.
(c) Use the tf-idf BoW representation.
