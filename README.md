# DEP
To build a natural language processing model for sentiment analysis, we'll use a combination of techniques including TF-IDF (Term Frequency-Inverse Document Frequency) and logistic regression.

First, let's start by importing the necessary libraries and loading our dataset
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

Next, we'll define a function to preprocess our text data
preprocessor = lambda text: re.sub(r'[^a-z ]', '', text.lower())
This function will remove any non-alphabetic characters and convert all text to lowercase

Now, let's load our dataset and split it into training and testing sets:
data = pd.read_csv('/PATH-TO-DATA/train.csv', names=['sentiment', 'title', 'review'])
X = data.review
y = data.sentiment.replace({1:'Negative', 2:'Positive'})
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

Next, we'll create a pipeline that includes TF-IDF transformation and logistic regression:
pipe = Pipeline([
    ('vec', CountVectorizer(stop_words='english', min_df=1000, preprocessor=preprocessor)),
    ('tfid', TfidfTransformer()),
    ('lr', SGDClassifier(loss='log'))
])

Finally, we can use our model to make predictions on the testing data and evaluate its performance:

python

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
This code will output a classification report that includes precision, recall, and F1 score for each class

