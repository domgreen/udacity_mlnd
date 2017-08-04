import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split

df = pd.read_table('~/python/udacity_mlnd/smsspamcollection/SMSSpamCollection',
                   sep='\t',
                   header=None,
                   names=['label', 'sms_message'])
df['label'] = df.label.map({'ham': 0, 'spam': 1})
x_train, x_test, y_train, y_test = train_test_split(df['sms_message'],
                                                    df['label'],
                                                    random_state=1)

print('Number of lines total: {}'.format(df.shape[0]))
print('Number of lines train: {}'.format(x_train.shape[0]))
print('Number of lines test : {}'.format(x_test.shape[0]))

count_vector = CountVectorizer()
# BoW - create the vocabulary and then transform it into a count of each word
training_data = count_vector.fit_transform(x_train)
testing_data = count_vector.transform(x_test)

print('BAYES THEORM')

# P(D)
p_diabetes = 0.01

# P('D)
p_no_diabetes = 0.99

# Sensitivity or P(Pos|D)
p_pos_diabetes = 0.9

# Specificity or P(Neg|'D)
p_neg_no_diabetes = 0.9

# P(Pos)
p_pos = (p_diabetes * p_pos_diabetes) + (p_no_diabetes * (1 - p_neg_no_diabetes))

print('The probability of getting a positive test result P(Pos) is: {}'.format(p_pos))

# P(D|Pos)
p_diabetes_pos = (p_diabetes * p_pos_diabetes) / p_pos

print('The probability of having diabetes given a positive test result: {}'.format(p_diabetes_pos))

# P(Pos|'D)
p_pos_no_diabetes = 0.1

# P('D|Pos)
p_no_diabetes_pos = (p_no_diabetes * p_pos_no_diabetes) / p_pos
print('The probability of NOT having diabetes given a positive test result: {}'.format(p_no_diabetes_pos))
