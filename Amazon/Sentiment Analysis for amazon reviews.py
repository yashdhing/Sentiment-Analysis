
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import nltk


# In[2]:


data = pd.read_csv("train.csv", encoding = 'latin')
print(data.head())


# In[3]:


data.dropna(inplace = True)
print(data.shape)
print(data['Sentiment'].mean())


# In[4]:


from sklearn.model_selection import train_test_split


# In[5]:


X_train, X_test, y_train, y_test = train_test_split(data['SentimentText'], data['Sentiment'], test_size=0.15, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.235, random_state=1)
print("X_train:", X_train.shape)
print("X_val:", X_val.shape)
print("X_test:", X_test.shape)


# In[6]:


from sklearn.feature_extraction.text import CountVectorizer


# In[7]:


countVect = CountVectorizer().fit(X_train)
print(countVect.get_feature_names()[::5000])
print("Feature count:", len(countVect.get_feature_names()))


# In[8]:


from sklearn.linear_model import LogisticRegression


# In[82]:


X_train_vectorized = countVect.transform(X_train)
model = LogisticRegression(C = 0.71)
model.fit(X_train_vectorized, y_train)


# In[83]:


predictions = model.predict(countVect.transform(X_val))


# In[7]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score


# In[84]:


print('AUC: ', roc_auc_score(y_val, predictions))
print("Accuracy score", accuracy_score(y_val, predictions))
print("f1_score(macro):", f1_score(y_val, predictions, average = 'macro'))
print("f1_score(micro):", f1_score(y_val, predictions, average = 'micro'))
print("f1_score(weighted):", f1_score(y_val, predictions, average = 'weighted'))


# In[85]:


feature_names = np.array(countVect.get_feature_names())

sorted_coef_index = model.coef_[0].argsort()

print('Smallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))
print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))


# In[14]:


import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|+*,#;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z @$_]')
STOPWORDS = set(stopwords.words('english'))

def text_prepare(text):
    """
        text: a string

        return: modified initial string
    """
    text = text.lower()
    text = re.sub(REPLACE_BY_SPACE_RE, " ", text)
    text = re.sub(BAD_SYMBOLS_RE, "", text)
    text_tokens = word_tokenize(text)
    filtered_sentence = ""
    for w in text_tokens:
        if w not in STOPWORDS:
            filtered_sentence += w + " "
    return filtered_sentence[:-1]


# In[15]:


X2_train = [text_prepare(x) for x in X_train]
X2_val = [text_prepare(x) for x in X_val]
X2_test = [text_prepare(x) for x in X_test]


# In[16]:


words_counts = {}

for tweet in X2_train:
    for word in tweet.split():
        words_counts[word] = words_counts.setdefault(word, 0) + 1

most_common_words = sorted(words_counts.items(), key=lambda x: x[1], reverse=True)[:10]

print(most_common_words)


# In[17]:


DICT_SIZE = 10000

most_common_words = sorted(words_counts.items(), key=lambda x: x[1], reverse=True)[:DICT_SIZE]

WORDS_TO_INDEX = {}
i = 0
for entry in most_common_words:
    WORDS_TO_INDEX[entry[0]] = i
    i += 1

INDEX_TO_WORDS = {}
i = 0
for entry in most_common_words:
    INDEX_TO_WORDS[i] = entry[0]
    i += 1

ALL_WORDS = WORDS_TO_INDEX.keys()

def my_bag_of_words(text, words_to_index, dict_size):
    result_vector = np.zeros(dict_size)
    for word in text.split():
        if word in words_to_index:
            result_vector[words_to_index[word]] += 1
    return result_vector


# In[18]:


from scipy import sparse as sp_sparse

X_train_mybag = sp_sparse.vstack([sp_sparse.csr_matrix(my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in X2_train])
X_val_mybag = sp_sparse.vstack([sp_sparse.csr_matrix(my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in X2_val])
X_test_mybag = sp_sparse.vstack([sp_sparse.csr_matrix(my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in X2_test])
print('X_train shape ', X_train_mybag.shape)
print('X_val shape ', X_val_mybag.shape)
print('X_test shape ', X_test_mybag.shape)


# In[98]:


classifier = LogisticRegression(C = 0.5).fit(X_train_mybag, y_train)


# In[99]:


predictions2 = classifier.predict(X_val_mybag)


# In[100]:


print('AUC2: ', roc_auc_score(y_val, predictions2))
print("Accuracy score", accuracy_score(y_val, predictions2))
print("f1_score(macro):", f1_score(y_val, predictions2, average = 'macro'))
print("f1_score(micro):", f1_score(y_val, predictions2, average = 'micro'))
print("f1_score(weighted):", f1_score(y_val, predictions2, average = 'weighted'))


# In[22]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[106]:


tfidfVect = TfidfVectorizer(min_df=5).fit(X_train)
print(len(tfidfVect.get_feature_names()))


# In[153]:


tfX_train_vectorized = tfidfVect.transform(X_train)

tfmodel = LogisticRegression(C = 1.2)
tfmodel.fit(tfX_train_vectorized, y_train)


# In[154]:


tfpredictions = tfmodel.predict(tfidfVect.transform(X_val))


# In[155]:


print('AUC2: ', roc_auc_score(y_val, tfpredictions))
print("Accuracy score", accuracy_score(y_val, tfpredictions))
print("f1_score(macro):", f1_score(y_val, tfpredictions, average = 'macro'))
print("f1_score(micro):", f1_score(y_val, tfpredictions, average = 'micro'))
print("f1_score(weighted):", f1_score(y_val, tfpredictions, average = 'weighted'))


# In[156]:


tffeature_names = np.array(tfidfVect.get_feature_names())

sorted_tfidf_index = tfX_train_vectorized.max(0).toarray()[0].argsort()

print('Smallest tfidf:\n{}\n'.format(tffeature_names[sorted_tfidf_index[:10]]))
print('Largest tfidf: \n{}'.format(tffeature_names[sorted_tfidf_index[:-11:-1]]))


# In[157]:


tfsorted_coef_index = tfmodel.coef_[0].argsort()

print('Smallest Coefs:\n{}\n'.format(tffeature_names[tfsorted_coef_index[:10]]))
print('Largest Coefs: \n{}'.format(tffeature_names[tfsorted_coef_index[:-11:-1]]))


# In[158]:


print(tfmodel.predict(tfidfVect.transform(['love',
                                    'working'])))


# In[159]:


ngramVect = CountVectorizer(min_df=5, ngram_range=(1,2)).fit(X_train)
print(len(ngramVect.get_feature_names()))


# In[166]:


ngramX_train_vectorized = ngramVect.transform(X_train)
ngramModel = LogisticRegression(C = 0.3)
ngramModel.fit(ngramX_train_vectorized, y_train)


# In[167]:


ngramPredictions = ngramModel.predict(ngramVect.transform(X_val))


# In[168]:


print('AUC2: ', roc_auc_score(y_val, ngramPredictions))
print("Accuracy score", accuracy_score(y_val, ngramPredictions))
print("f1_score(macro):", f1_score(y_val, ngramPredictions, average = 'macro'))
print("f1_score(micro):", f1_score(y_val, ngramPredictions, average = 'micro'))
print("f1_score(weighted):", f1_score(y_val, ngramPredictions, average = 'weighted'))


# In[37]:


ngramfeature_names = np.array(ngramVect.get_feature_names())

ngramsorted_coef_index = ngramModel.coef_[0].argsort()

print('Smallest Coefs:\n{}\n'.format(ngramfeature_names[ngramsorted_coef_index[:10]]))
print('Largest Coefs: \n{}'.format(ngramfeature_names[ngramsorted_coef_index[:-11:-1]]))


# In[38]:


print(ngramModel.predict(ngramVect.transform(['phone is good',
                                    'an issue, phone is not working', 'horrible phone'])))


# In[8]:


from sklearn import svm


# In[9]:


clfVect = CountVectorizer(min_df=5).fit(X_train)


# In[10]:


clfX_train_vectorized = clfVect.transform(X_train) 


# In[ ]:


clf = svm.SVC()
clf.fit(clfX_train_vectorized, y_train)
clfpredictions = clf.predict(clfVect.transform(X_val))


# In[ ]:


print('AUC2: ', roc_auc_score(y_val, clfpredictions))
print("Accuracy score", accuracy_score(y_val, clfpredictions))
print("f1_score(macro):", f1_score(y_val, clfpredictions, average = 'macro'))
print("f1_score(micro):", f1_score(y_val, clfpredictions, average = 'micro'))
print("f1_score(weighted):", f1_score(y_val, clfpredictions, average = 'weighted'))

