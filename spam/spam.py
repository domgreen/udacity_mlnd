import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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

naive_bayes = MultinomialNB()
naive_bayes.fit(training_data, y_train)


'''
Instructions:
Now that our algorithm has been trained using the training data set we can now make some predictions on the test data
stored in 'testing_data' using predict(). Save your predictions into the 'predictions' variable.
'''

predictions = naive_bayes.predict(testing_data)

'''
Instructions:
Compute the accuracy, precision, recall and F1 scores of your model using your test data 'y_test' and the predictions
you made earlier stored in the 'predictions' variable.
'''

print('Accuracy Score: ', format(accuracy_score(y_test, predictions)))
print('Precision Score: ', format(precision_score(y_test, predictions)))
print('Recall Score: ', format(recall_score(y_test, predictions)))
print('F1 Score: ', format(f1_score(y_test, predictions)))
