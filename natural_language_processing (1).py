
# coding: utf-8




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt





cd Desktop/





df=pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting=3) #Convert the above file to .tsv for better results.




import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus=[]
for i in range(0,1000):
    review=re.sub('[^a-zA-Z]',' ',df['Review'][i])
    review=review.lower()
    review=review.split()
    ps=PorterStemmer()
    review=[ps.stem(word) for word in review  if not word in set(stopwords .words('english'))]
    review=' '.join (review)
    corpus.append(review)

print(corpus)






from sklearn.feature_extraction.text import CountVectorizer




cv=CountVectorizer(max_features=1500)





x=cv.fit_transform(corpus).toarray()
x





y=df.iloc[:,1].values





print(y)


# This is the sparse matrix for the above queation.




from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)#Split the data into training set and test set.





from sklearn.naive_bayes import GaussianNB
cf=GaussianNB()
cf.fit(x_train,y_train)




y_predict=cf.predict(x_test)





from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_predict) #Create confusion matrix





print(cm) #print confusion matrix.

#Once we print the confusion matrix we get it as follows.
array([[55, 42],
       [12, 91]])
# So,according to confusion matrix,total correct predictions = 55+91=156.
and incorrect predictions=12+42=54
# accuracy=156/200
# =76 percent.

