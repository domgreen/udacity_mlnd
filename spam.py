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

frequency_list = []
for i in documents:
    sans_punctiation = i.lower().translate(str.maketrans('', '', string.punctuation))
    frequency_count = Counter(sans_punctiation.split(' '))
    frequency_list.append(frequency_count)

print(frequency_list)
