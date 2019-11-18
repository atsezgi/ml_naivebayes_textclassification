# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 20:24:30 2019

@author: ezgi
"""

import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split 
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.classify import SklearnClassifier
import re
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix 

df_result = pd.read_csv('train.csv') #read the data set
df_result = df_result[['Insult','Comment']] #use the insult and comment rows for the classification
df_pos = df_result[df_result.Insult==1] #select the innsult is equal to 1
df_neg = df_result[df_result.Insult ==0] #select the insult is equal to 0

frames =[df_pos, df_neg] 
dfa = pd.concat(frames, axis=0) #verimzin son hali

df = dfa.sample(n=1000) #bütün veriden rasgele 1000 sample seçiyoruz
value_count=df.Insult.value_counts()
ax = sns.countplot(x="Insult", data=dfa,palette="Set3") # Verimizde hangi sınıftan kaçar adet olduğunu görselleştirme

#this function remove the numbers from the comment row
def remove_num(text):
        no_num = re.sub('\w*\d\w*', '', text)
        return no_num
df['text_no_num'] = df['Comment'].apply(lambda x: remove_num(x)) #apply the reomove_num func. to comment row

#this function the urls if there is from the comment row
def remove_url(text):
    text_nourl = re.sub('https?://[A-Za-z0-9./]+','',text)
    return text_nourl
df['text_nourl'] = df['text_no_num'].apply(lambda x:remove_url(x))

#this function remeove the mentions if there is, the dataset can be twitter dataset
def remove_mention(text):
    no_ment = re.sub(r'@[A-Za-z0-9]+','',text)
    return no_ment
df['text_noment'] = df['text_nourl'].apply(lambda x:remove_mention(x))

#this function remove the punctuation from the comment row
import string
def remove_punct(text):
    text_nonpunct = "".join([char for char in text if char not in string.punctuation])
    return text_nonpunct

df['text_nopunc'] = df['text_noment'].apply(lambda x: remove_punct(x))

#this function tokenize the Commnent row
wpt = nltk.WordPunctTokenizer()
def tokenizer(text):
    tokens = wpt.tokenize(text)
    return tokens
df['text_tokenized'] = df['text_nopunc'].apply(lambda x: tokenizer(x.lower()))

#this function remove the stopwords
stopword = nltk.corpus.stopwords.words('english')
def remove_stopwords(tokenized_list):
    text = [word for word in tokenized_list if word not in stopword]
    return text



x = df['text_nopunc'].values #Independent variable
y=df['Insult'].values #Dependent variable

#Visualize the insult and not insult words with wordCloud
from wordcloud import WordCloud
neg_comments = df[df.Insult==1]
neg_strings = []
for t in neg_comments.text_nopunc:
    neg_strings.append(t)
neg_strings = pd.Series(neg_strings).str.cat(sep=' ')

wordcloud = WordCloud(width=1600, height=800,max_font_size=200,colormap='spring').generate(neg_strings)
plt.figure(figsize=(12,10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.title("Visualization of Insult Words")
plt.axis("off")
plt.show()


pos_comments =df[df.Insult==0]
pos_strings = []
for t in pos_comments.text_nopunc:
    pos_strings.append(t)
pos_strings= pd.Series(pos_strings).str.cat(sep=' ')

wordcloud = WordCloud(width=1600, height=800,max_font_size=200,colormap='Set2').generate(pos_strings)
plt.figure(figsize=(12,10))
plt.imshow(wordcloud, interpolation="bilinear") 
plt.title("Visualization of Not Insult Words")
plt.axis("off") 
plt.show()



from sklearn.feature_extraction.text import CountVectorizer
# Convert a collection of text documents to a matrix of token counts
count_vect = CountVectorizer()
x_counts = count_vect.fit_transform(x).toarray()
print(x_counts.shape)




#split the dataset train(80%) and test(20%)
x_train, x_test, y_train, y_test = train_test_split(x_counts,y,test_size=0.2, random_state=420)
from sklearn.naive_bayes import GaussianNB

#use gaussian naive bayes
lr = GaussianNB()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)

#confusion matrix
cm=confusion_matrix(y_test, lr.predict(x_test))
print(cm)
print(classification_report(y_test, lr.predict(x_test)))

#visualize the confusion matrix
plt.clf()
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Pastel1)
classNames = ['Negative','Positive']
plt.title('Confusion Matrix Test Data')
plt.ylabel('True label')
plt.xlabel('Predicted label')
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation=45)
plt.yticks(tick_marks, classNames)
s = [['TN','FP'], ['FN', 'TP']]
for i in range(2):
    for j in range(2):
        plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
plt.show()















