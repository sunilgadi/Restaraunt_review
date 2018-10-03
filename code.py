import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

dataset = pd.read_csv('../input/Restaurant_Reviews.tsv' , delimiter = '\t' , quoting=3)

review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][0]) # Doing only for 1st review
review = review.lower() # Change to lower case
review = review.split() # Break the string to a list of words 

import nltk
# nltk.download('stopwords') ---------- To download the stopwords package
from nltk.corpus import stopwords # importy stopwords package
review = [word for word in review if not word in set(stopwords.words('english'))] 

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
review = [ps.stem(word) for word in review] # This can be clubbed with the above for loop when removing stop words

review = " ".join(review) # Join the words to form a string

corpus = [] # Save in this list
for i in range(dataset.shape[0]):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = " ".join(review)
    corpus.append(review)
    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500) # Here max_features will only take the most frequent 1500 words occuring, by doing this we remove only once occuring words
X = cv.fit_transform(corpus).toarray()

y = dataset.iloc[:,-1].values

from sklearn.cross_validation import train_test_split
X_train , X_test, y_train , y_test = train_test_split(X , y , test_size = 0.2 , random_state = 0)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test , y_pred)
