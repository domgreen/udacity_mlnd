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

print('NAIVE BAYES THEORM')

'''
Instructions: Compute the probability of the words 'freedom' and 'immigration' being said in a speech, or
P(F,I).

The first step is multiplying the probabilities of Jill Stein giving a speech with her individual
probabilities of saying the words 'freedom' and 'immigration'. Store this in a variable called p_j_text
'''

# P(J)
p_j = 0.5

# P(F|J)
p_j_f = 0.1

# P(I|J)
p_j_i = 0.1

p_j_text = p_j * p_j_f * p_j_i
print(p_j_text)

'''
The second step is multiplying the probabilities of Gary Johnson giving a speech with his individual
probabilities of saying the words 'freedom' and 'immigration'. Store this in a variable called p_g_text
'''

# P(G)
p_g = 0.5

# P(F|G)
p_g_f = 0.7

# P(I|G)
p_g_i = 0.2

p_g_text = p_g * p_g_f * p_g_i

print(p_g_text)

'''
The third step is to add both of these probabilities and you will get P(F,I).
'''

p_f_i = p_g_text + p_j_text
print('Probability that words Freedom and Immigration are said: {}'.format(p_f_i))

'''
Instructions:
Compute P(J|F,I) using the formula P(J|F,I) = (P(J) * P(F|J) * P(I|J)) / P(F,I) and store it in a variable p_j_fi
'''

# P(J|F,I)
p_j_fi = (p_j_text) / p_f_i
print('The probability of Jill given words Freedom and Imigration: {}'.format(p_j_fi))

# P(G|F,I)
p_g_fi = (p_g_text) / p_f_i
print('The probability of Gary given words Freedom and Immigration: {}'.format(p_g_fi))
