#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

nltk.download('stopwords')

# Load the data
data = pd.read_csv("Train.csv")
print(data.head())

print(data.shape)
data = data.iloc[:1000, :]
print(data.shape)

# Working with Label
print(data['label'].value_counts())
data['label'].value_counts().plot(kind='bar')

plt.figure(figsize=(10, 6))
colors = ['green', 'orange']
data['label'].value_counts().plot(kind='pie', autopct='%.1f%%', shadow=True, colors=colors, startangle=45, explode=(0, 0.1))
plt.title('Label Distribution')
plt.show()

# Working with Text
print(data['text'][999])

# Cleaning steps
'''
- Removing HTML Tags
- Extracting emojis
- Removing special characters, punctuation, symbols
- Converting to lowercase
- Removing stopwords
- Tokenization
'''

stopwords_set = set(stopwords.words('english'))
emoji_pattern = re.compile(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)')

def preprocessing(text):
    # Removing HTML tags
    text = re.sub('<[^>]*>', '', text)
    # Extracting and appending emojis
    emojis = ' '.join(emoji_pattern.findall(text))
    # Removing special characters and converting to lowercase
    text = re.sub(r'[\W_]+', ' ', text.lower()) + ' ' + emojis.replace('-', '')
    # Tokenization and stemming
    prter = PorterStemmer()
    text = [prter.stem(word) for word in text.split() if word not in stopwords_set]

    return " ".join(text)

# Example of preprocessing
print(preprocessing('This is my tags <h1> :) <p>Hello World<p> <div> <div> </h2>'))


data['text'] = data['text'].apply(lambda x: preprocessing(x))
print(data['text'])




#Visualizing Negative and Positive Words


positivedata = data[data['label'] == 1]
positivedata = positivedata['text']
negdata = data[data['label'] == 0]
negdata = negdata['text']

import matplotlib.pyplot as plt
from collections import Counter

# Positive data
positivedata_words = ' '.join(positivedata).split()
positivedata_word_counts = Counter(positivedata_words)
positivedata_common_words = positivedata_word_counts.most_common(10)  # Display top 10 common words

# Negative data
negdata_words = ' '.join(negdata).split()
negdata_word_counts = Counter(negdata_words)
negdata_common_words = negdata_word_counts.most_common(10)  # Display top 10 common words

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Positive data word frequency
axes[0].barh(range(len(positivedata_common_words)), [count for _, count in positivedata_common_words], align='center')
axes[0].set_yticks(range(len(positivedata_common_words)))
axes[0].set_yticklabels([word for word, _ in positivedata_common_words])
axes[0].set_title('Positive Data Word Frequency')

# Negative data word frequency
axes[1].barh(range(len(negdata_common_words)), [count for _, count in negdata_common_words], align='center')
axes[1].set_yticks(range(len(negdata_common_words)))
axes[1].set_yticklabels([word for word, _ in negdata_common_words])
axes[1].set_title('Negative Data Word Frequency')

plt.tight_layout()
plt.show()


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf=TfidfVectorizer(strip_accents=None,lowercase=False,preprocessor=None,use_idf=True,norm='l2',smooth_idf=True)
y=data.label.values
x=tfidf.fit_transform(data.text)

#training the model

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,random_state=1,test_size=0.5,shuffle=False)

from sklearn.linear_model import LogisticRegressionCV
clf=LogisticRegressionCV(cv=6,scoring='accuracy',random_state=0,n_jobs=-1,verbose=3,max_iter=500).fit(X_train,y_train)
y_pred = clf.predict(X_test)


#Accuracy
from sklearn import metrics

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

import pickle
pickle.dump(clf,open('clf.pkl','wb'))
pickle.dump(tfidf,open('tfidf.pkl','wb'))

