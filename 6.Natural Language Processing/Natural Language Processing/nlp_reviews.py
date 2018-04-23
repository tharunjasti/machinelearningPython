# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

#nltk.download('stopwords')
#ntlk.downloads('all')

dataset=pd.read_csv( 'Restaurant_Reviews.tsv', delimiter='\t', quoting=3)
reviews=[]
#range(0, len(dataset['Review'])
#cleaning text words i.e, remove puncuations,removed stopwords, replace with stem words
for i in range(0, len(dataset['Review']) ) :
    cleanText=dataset['Review'][i]
    cleanText=re.sub('[^a-zA-Z]',' ',cleanText)
    cleanText=cleanText.lower()
    cleanText=cleanText.split()
    ps=PorterStemmer()
    cleanText=[ps.stem(word) for word in cleanText if word not in stopwords.words('english')]
    cleanText=' '.join(cleanText)
    reviews.append(cleanText)

  
# Adding bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
X=cv.fit_transform(reviews).toarray()
y=dataset['Liked'].values

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.20, random_state=0)

#predicting with model
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
model=GaussianNB()

#model=RandomForestClassifier(n_estimators=1000, criterion='entropy')
#model=SVC(kernel='rbf', random_state=0)
model.fit(X_train,y_train)
pred=model.predict(X_test)

#metrics
from sklearn.metrics import confusion_matrix, classification_report
cm=confusion_matrix(y_test,pred)
report=classification_report(y_test,pred)