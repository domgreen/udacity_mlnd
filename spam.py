import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_table('~/python/udacity_mlnd/smsspamcollection/SMSSpamCollection',
                   sep='\t',
                   header=None,
                   names=['label', 'sms_message'])
df['label'] = df.label.map({'ham': 0, 'spam': 1})

print(df.shape)
print(df.head())

documents = ['Hello, how are you!',
             'Win money, win from home.',
             'Call me now.',
             'Hello, Call hello you tomorrow?']

count_vector = CountVectorizer()
count_vector.fit(documents)
doc_array = count_vector.transform(documents).toarray()
frequency_matrix = pd.DataFrame(doc_array,
                                columns=count_vector.get_feature_names())
print(frequency_matrix)
