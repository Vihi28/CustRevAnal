
import pickle
import pandas as pd

df = pd.read_csv('sent.csv')

logreg = pickle.load(open('model_save.sav', 'rb'))
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=1000)
vectors = vectorizer.fit_transform(df.text)
words_df = pd.DataFrame(vectors.toarray(), columns=vectorizer.get_feature_names())

print(vectorizer.get_feature_names)
from flask import Flask, render_template, request

app = Flask(__name__,template_folder="")

@app.route("/")
def home():
    return render_template("w1.html")

@app.route("/w2")
def wp2():
    return render_template("w2.html")

@app.route("/w3")
def wp3():
    rev = request.args.get("rev")
    pd.set_option("display.max_colwidth", 200)
    unknown = pd.DataFrame({'content': [
        rev
    ]})
    unknown_vectors = vectorizer.transform(unknown.content)
    unknown_words_df = pd.DataFrame(unknown_vectors.toarray(), columns=vectorizer.get_feature_names())

    u = logreg.predict(unknown_words_df)
    if(u[0]==0):
        pred = "Negative"
    else:
        pred = "Positive"
    return render_template("w3.html", rev = rev, pred = pred)

if __name__ == "__main__":
    app.run()
