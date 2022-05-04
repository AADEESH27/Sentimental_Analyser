from flask import Flask, render_template, request
#from nltk.sentiment.vader import SentimentIntensityAnalyzer
#import nltk
#nltk.download('vader_lexicon')
import pandas as pd
import numpy as np
import preprocess_kgptalkie as ps

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report


app = Flask(__name__)


@app.route('/', methods=["GET", "POST"])
def main():
    if request.method == "POST":
        z = request.form.get("inp")
        score = sentiment_analysis(z)
        if score == 0:
            return render_template('Ex1.html', message="Positive😊😊😊")
        elif score == 1:
            return render_template('Ex1.html', message="Negative😢😢😢")
    return render_template('Ex1.html')



def sentiment_analysis(z):
    df = pd.read_csv('https://raw.githubusercontent.com/AADEESH27/Sentimental_Analyser/main/twitter-suicidal_data.csv')
    df.head()
    df['intention'].value_counts()
    df['tweet'] = df['tweet'].apply(lambda x: get_clean(x))
    df.head()
    tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 3), analyzer='char')

    X = tfidf.fit_transform(df['tweet'])
    y = df['intention']

    X.shape

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    clf = LinearSVC()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print(classification_report(y_test, y_pred))


    z = get_clean(z)
    vec = tfidf.transform([z])
    a = int(clf.predict(vec))
    return a

def get_clean(x):
    x = str(x).lower().replace('\\', '').replace('_', ' ')
    x = ps.cont_exp(x)
    x = ps.remove_emails(x)
    x = ps.remove_urls(x)
    x = ps.remove_html_tags(x)
    x = ps.remove_rt(x)
    x = ps.remove_accented_chars(x)
    x = ps.remove_special_chars(x)
    x = re.sub("(.)\\1{2,}", "\\1", x)
    return x


