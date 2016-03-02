import string
import numpy as np
from datetime import datetime
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import CountVectorizer 
from os import listdir 
from os.path import isfile
from nltk.corpus import stopwords
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 

WNL=WordNetLemmatizer()
#to list all the files in the directory and also label them spam =1 and ham=0
def all_files(mypath):
    part=[]
    labels=[]
    i=0
    l=0
    #mypath= "F:\sem8\cs771a\ass2\bare"
    for dirname in listdir(mypath):
        ftemp=[]
        ltemp=[]
        for filename in listdir(mypath+"\\"+dirname):
            if isfile(mypath+"\\"+dirname+"\\"+filename):
                ftemp.append(mypath+"\\"+dirname+"\\"+filename)
                if filename.startswith('spm'):
                    l=1
                else:
                    l=0
            ltemp.append(l)
        labels.append(ltemp)
        part.append(ftemp)
    label_file=open("label_c.txt","a")
    label_file.write(str(datetime.now()))
    label_file.write('\n')
    label_file.write(str(labels))
    label_file.write('\n')
    print("labels saved in label_c.txt")
    label_file.close()
    return part, labels

###############################
#to clear emails from punctuations, numbers and first letter(i.e. Subject)
def clear_all_emails(part):     #return clean_data which contains only word string of all the emails
    clean_data=[]
    for x in range(len(part)):
        clean=[]
        print("part %d in progress\n"%(x))
        for y in range(len(part[x])):
           
            rawdata=open(part[x][y]).read()
            for c in string.punctuation :
                rawdata=rawdata.replace(c,"")
            fresh_data=[]
            i=0;
            for word in rawdata.split():
                if i==0:                       #to remove first letter i.e. Subject
                    i=i+1
                elif not word.isdigit():       #to remove all numbers and lemmatize simultaneously
                    fresh_data.append(WNL.lemmatize(word))

            fresh_data=" ".join([letter for letter in fresh_data if letter not in stopwords.words("english")])
           
            clean.append(fresh_data)


        clean_data.append(clean)
    return clean_data
#################################

mypath="F:\\sem8\\cs771a\\ass2\\bare"    #specify the name of the directory bare
[filenames,label]=all_files(mypath)      #email database file names
print("no of parts=",len(filenames))
#clear all mails 
clear_data=clear_all_emails(filenames)    
gnb=GaussianNB()
prediction=np.zeros([len(filenames),1])
pre_file=open("prediction_c.txt","a")
pre_file.write(str(datetime.now()))
pre_file.write('\n')
for test in range(len(filenames)):  # 10 fold cross validation 
    train_data=[]
    test_data=[]
    train_result=[]
    test_result=[]
    for k in range(len(filenames)):         #segregating train and test data
        if k==test:
            test_data.extend(clear_data[k])
            test_result.extend(label[k])
        else:
            train_data.extend(clear_data[k])      
            train_result.extend(label[k])
    
    ## credit :https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-1-for-beginners-bag-of-words
    # initialise CountVectorizer
    vectorizer = CountVectorizer(analyzer = "word",   
                                 tokenizer = None,    
                                 preprocessor = None, 
                                 stop_words = None,
                                 binary=True) 

    # fit_transform() does two functions: First, it fits the model
    # and learns the vocabulary; second, it transforms our training data
    # into feature vectors. The input to fit_transform should be a list of 
    # strings.
    train_data_features = vectorizer.fit_transform(train_data)
    train_data_features = train_data_features.toarray()
    test_data_features = vectorizer.transform(test_data)
    test_data_features = test_data_features.toarray()
    gnb.fit(train_data_features,train_result)
    pre=gnb.predict(test_data_features)
    pre=np.array(pre)
    test_result=np.array(test_result)
    pre_file.write("mismatch ="+str((pre!=test_result).sum())+"out of " +str(len(test_result))+"\n")
    prediction[test]=f1_score(test_result,pre,average='binary')
print(prediction.sum()/10)   

pre_file.write(str(prediction))
pre_file.write('\n')
pre_file.write(str(prediction.sum()/10))
pre_file.write('\n')
print("prediction saved in prediction_c.txt")
pre_file.close()