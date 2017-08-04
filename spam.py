import pandas as pd

df = pd.read_table('~/python/udacity_mlnd/smsspamcollection/SMSSpamCollection',
                   sep='\t',
                   header=None,
                   names=['label', 'sms_message'])

print(df.head())
