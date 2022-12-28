from flask import Flask, render_template, request, flash

app = Flask(__name__)
app.secret_key = "manbearpig_MUDMAN888"
import json
import requests
import re


from PIL import Image
import base64
import io

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#text processing & sentiment analysis
import re
from nltk.stem import WordNetLemmatizer
import nltk
#nltk.download('stopwords')
from wordcloud import WordCloud
##from afinn import Afinn
import unicodedata as ud
##import camel_tools as ct
from nltk.stem.isri import ISRIStemmer
##from ar_wordcloud import ArabicWordCloud
import time
#model
from textblob import TextBlob
##from camel_tools.sentiment import SentimentAnalyzer
from sklearn.metrics import classification_report, accuracy_score

from bidi.algorithm import get_display
import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import arabic_reshaper # this was missing in your code

# os.chdir("C:")


import nltk
##from nltk.corpus import stopwords

#stopwords = set(stopwords.words('arabic'))




def wrdcld(text):
    processedtext_ar = preprocess_ar(text)
    awc = ArabicWordCloud(background_color="white")
    plt.figure(figsize = (16,16))
    wc_ar = awc.from_text(u''.join(processedtext_ar))
    image=plt.imshow(wc_ar)
    return (image)



headers = {"Authorization": f"Bearer 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"}
API_URL = "https://api-inference.huggingface.co/models/Ammar-alhaj-ali/arabic-MARBERT-sentiment"
API_URL1 = "https://api-inference.huggingface.co/models/CAMeL-Lab/bert-base-arabic-camelbert-mix-sentiment"
API_URL2 = "https://api-inference.huggingface.co/models/CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment"
API_URL3 = "https://api-inference.huggingface.co/models/Yah216/Sentiment_Analysis_CAMelBERT_msa_sixteenth_HARD"

##from tabulate import tabulate
import json
import pandas as pd
from collections import Counter
from facebook_scraper import get_profile,get_posts, get_friends
import re     
import time
##pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', None)  # or 199s
def get_comment (account_name):
    comments=[]
    for post in get_posts(account_name, pages=100, timeout=60, options={"comments" : "generator", "allow_extra_requests": True,"progress": True},cookies="facebook.com_cookies.txt" ):
        for comment in post["comments_full"]:
                comments.append(comment["comment_text"])
                ##for reply in comment["replies"]:
                    ##comments.append(reply["comment_text"])
                    ##print(reply)
                    ##if len(comments)>1000:
                        ##break
                if len(comments)>1000:
                    break
        if len(comments)>1000:
                    break
    print(len(comments))
    return(comments)







def list2json (text,scor,sent):
    d = [ { 'text': x, 'score': y, 'sentiment': z } for x, y, z in zip(text, scor, sent) ]
    pretty_json = json.dumps(d, sort_keys=True, indent=4,ensure_ascii=False)
    return pretty_json
    

def query(payload):
    data = json.dumps(payload)
    response = requests.request("POST", API_URL3, headers=headers, data=data)
    return json.loads(response.content.decode("utf-8"))

@app.route("/hello")
def index():
	flash("Please Select Function!!!!")
	return render_template("index.html")

@app.route("/greet", methods=['POST', 'GET'])
def greeter1():
        return render_template("sentimentoftext.html")
@app.route("/sentiment", methods=['POST', 'GET'])
def greeter2():
        data2 = query({"inputs": request.form['name_input']})
        data=str(data2).split(",")
        data=data[0]
        data=str(data).split(":")
        data=data[1]
        data=re.sub("'","",str(data))
        flash(data)
        return render_template("sentimentoftext.html")
@app.route("/retoursentiment", methods=['POST', 'GET'])
def greeter3():
	return render_template("index.html")

@app.route("/greet2", methods=['POST', 'GET'])
def greeter4():
	###flash("Hi " + str(request.form['name_input1']) + ", great to see you!")
	return render_template("SBK.html")

@app.route("/greet3", methods=['POST', 'GET'])
def greeter5():
	###flash("Hi " + str(request.form['name_input1']) + ", great to see you!")
	return render_template("SFP.html")


@app.route("/SBK", methods=['POST', 'GET'])
def greeter6():
	###flash("Hi " + str(request.form['name_input1']) + ", great to see you!")
	return render_template("SBK.html")

@app.route("/retourSBK", methods=['POST', 'GET'])
def greeter7():
	###flash("Hi " + str(request.form['name_input1']) + ", great to see you!")
	return render_template("index.html")
###sa = SentimentAnalyzer("CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment")

@app.route("/SFP", methods=['POST', 'GET'])
def greeter8():
      label1=[]
      t = time.time()
      c=get_comment(account_name=request.form['name_input3'])
      ###allscore=sa.predict(sentences)
      allscore = query({"inputs": c})
      flash(allscore)
      score1=[]
      ####try:
      ff1=''
      ff2=''
      ff3=''
      ff4=''
      ff5=''
      ff6=''
      for j in allscore:
         allscoresplit=str(j).split(",")
         label=allscoresplit[0]
         ###print(str(allscoresplit))
         score=allscoresplit[1]
         score=re.sub(" 'score': ","",score)
         score=re.sub("}","",score)
         score=round(float(score), 3)
         score1.append(str(score))
         label=re.sub("\\[{'label': '","",label)
         label=re.sub("'","",label)
         label1.append(label)
      ##flash(Counter(label1))
      df=pd.DataFrame(list(zip(c, score1, label1)),columns=['text','scoore1', 'label1'])
      df=df.sort_values(by = ['label1','scoore1'],ascending=False)
      print(df)
      df1 = df[df['label1'] == 'positive' ]
      df2 = df[df['label1'] == 'negative' ]
      ff1=df1['text'].values[0]
      ff2=df1['text'].values[1]
      ff3=df1['text'].values[2]
      ff4=df2['text'].values[0]
      ff5=df2['text'].values[1]
      ff6=df2['text'].values[2]
      text_ar= list(df1["text"])
      data = arabic_reshaper.reshape(text_ar)
      data = get_display(data) # add this line
      WordCloud = WordCloud(font_path='arial', background_color='white',mode='RGB', width=2000, height=1000).generate(data)
      im = Image.open(WordCloud)
      data = io.BytesIO()
      im.save(data, "JPEG")
      encoded_img_data = base64.b64encode(data.getvalue())
##          plt.imshow(WordCloud)
##          plt.axis("off")
##          plt.show()
##          flash(WordCloud)
      tiiii=time.time()-t
      flash(tiiii)
      ##except Exception as e:
               ###flash(e)
      return render_template("SFP.html",v1=ff4,v2=ff5,v3=ff6,v4=ff1,v5=ff2,v6=ff3,wrd=encoded_img_data.decode('utf-8'))
@app.route("/retourSFP", methods=['POST', 'GET'])
def greeter9():
	###flash("Hi " + str(request.form['name_input1']) + ", great to see you!")
	return render_template("index.html")
