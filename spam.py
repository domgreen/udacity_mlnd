import pandas as pd
import string
from collections import Counter

#df = pd.read_table('~/python/udacity_mlnd/smsspamcollection/SMSSpamCollection',
#                   sep='\t',
#                   header=None,
#                   names=['label', 'sms_message'])
#df['label'] = df.label.map({'ham': 0, 'spam': 1})

#print(df.shape)
#print(df.head())

documents = ['Hello, how are you!',
             'Win money, win from home.',
             'Call me now.',
             'Hello, Call hello you tomorrow?']

lower_case_documents = []
for i in documents:
    lower_case_documents.append(i.lower())

print(lower_case_documents)

sans_punctuation = []

for i in lower_case_documents:
    sans_punctuation.append(i.translate(str.maketrans('', '', string.punctuation)))

print(sans_punctuation)

preprocessed_documents = []
for i in sans_punctuation:
    preprocessed_documents.append(i.split(' '))

print(preprocessed_documents)

frequency_list = []
for i in preprocessed_documents:
    frequency_count = Counter(i)
    frequency_list.append(frequency_count)

print(frequency_list)

