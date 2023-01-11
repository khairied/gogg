from flask import Flask, render_template, request, flash
from flask import Flask, render_template, request, redirect, url_for, session
#from flask_mysqldb import MySQL
#import MySQLdb.cursors
import re
app = Flask(__name__)

# Change this to your secret key (can be anything, it's for extra protection)
#app.secret_key = 'your secret key'

# Enter your database connection details below
#app.config['MYSQL_HOST'] = 'localhost'
#app.config['MYSQL_USER'] = 'root'
#app.config['MYSQL_PASSWORD'] = ''
#app.config['MYSQL_DB'] = 'pythonlogin'

# Intialize MySQL
#mysql = MySQL(app)
app = Flask(__name__)
app.secret_key = "manbearpig_MUDMAN888"
import json
import requests
import re
import gradio as gr 

from PIL import Image
import base64
import io
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#text processing & sentiment analysis
import re
from nltk.stem import WordNetLemmatizer
import nltk
from wordcloud import WordCloud
import unicodedata as ud
from nltk.stem.isri import ISRIStemmer
from ar_wordcloud import ArabicWordCloud
import time
#model
from textblob import TextBlob
from sklearn.metrics import classification_report, accuracy_score

from bidi.algorithm import get_display
import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import arabic_reshaper # this was missing in your code
import json
from collections import Counter
from facebook_scraper import get_profile,get_posts, get_friends
import re     
import time
from nltk.tokenize import word_tokenize
import nltk


stopwords = ['إذ', 'إذا', 'إذما', 'إذن', 'أف', 'أقل', 'أكثر', 'ألا', 'إلا', 'التي', 'الذي', 'الذين', 'اللاتي', 'اللائي', 'اللتان', 'اللتيا', 'اللتين', 'اللذان', 'اللذين', 'اللواتي', 'إلى', 'إليك', 'إليكم', 'إليكما', 'إليكن', 'أم', 'أما', 'أما', 'إما', 'أن', 'إن', 'إنا', 'أنا', 'أنت', 'أنتم', 'أنتما', 'أنتن', 'إنما', 'إنه', 'أنى', 'أنى', 'آه', 'آها', 'أو', 'أولاء', 'أولئك', 'أوه', 'آي', 'أي', 'أيها', 'إي', 'أين', 'أين', 'أينما', 'إيه', 'بخ', 'بس', 'بعد', 'بعض', 'بك', 'بكم', 'بكم', 'بكما', 'بكن', 'بل', 'بلى', 'بما', 'بماذا', 'بمن', 'بنا', 'به', 'بها', 'بهم', 'بهما', 'بهن', 'بي', 'بين', 'بيد', 'تلك', 'تلكم', 'تلكما', 'ته', 'تي', 'تين', 'تينك', 'ثم', 'ثمة', 'حاشا', 'حبذا', 'حتى', 'حيث', 'حيثما', 'حين', 'خلا', 'دون', 'ذا', 'ذات', 'ذاك', 'ذان', 'ذانك', 'ذلك', 'ذلكم', 'ذلكما', 'ذلكن', 'ذه', 'ذو', 'ذوا', 'ذواتا', 'ذواتي', 'ذي', 'ذين', 'ذينك', 'ريث', 'سوف', 'سوى', 'شتان', 'عدا', 'عسى', 'عل', 'على', 'عليك', 'عليه', 'عما', 'عن', 'عند', 'غير', 'فإذا', 'فإن', 'فلا', 'فمن', 'في', 'فيم', 'فيما', 'فيه', 'فيها', 'قد', 'كأن', 'كأنما', 'كأي', 'كأين', 'كذا', 'كذلك', 'كل', 'كلا', 'كلاهما', 'كلتا', 'كلما', 'كليكما', 'كليهما', 'كم', 'كم', 'كما', 'كي', 'كيت', 'كيف', 'كيفما', 'لا', 'لاسيما', 'لدى', 'لست', 'لستم', 'لستما', 'لستن', 'لسن', 'لسنا', 'لعل', 'لك', 'لكم', 'لكما', 'لكن', 'لكنما', 'لكي', 'لكيلا', 'لم', 'لما', 'لن', 'لنا', 'له', 'لها', 'لهم', 'لهما', 'لهن', 'لو', 'لولا', 'لوما', 'لي', 'لئن', 'ليت', 'ليس', 'ليسا', 'ليست', 'ليستا', 'ليسوا', 'ما', 'ماذا', 'متى', 'مذ', 'مع', 'مما', 'ممن', 'من', 'منه', 'منها', 'منذ', 'مه', 'مهما', 'نحن', 'نحو', 'نعم', 'ها', 'هاتان', 'هاته', 'هاتي', 'هاتين', 'هاك', 'هاهنا', 'هذا', 'هذان', 'هذه', 'هذي', 'هذين', 'هكذا', 'هل', 'هلا', 'هم', 'هما', 'هن', 'هنا', 'هناك', 'هنالك', 'هو', 'هؤلاء', 'هي', 'هيا', 'هيت', 'هيهات', 'والذي', 'والذين', 'وإذ', 'وإذا', 'وإن', 'ولا', 'ولكن', 'ولو', 'وما', 'ومن', 'وهو', 'يا', 'يمالى', 'قل', 'كثر', 'ألي', 'ليك', 'ليكم', 'نتيا', 'نتوما',  'بصح', 'هوما', 'وين', 'أمبعد', 'أومبعد', 'شوية', 'شويا', 'وش', 'واش', 'بوش', 'بواش']
st_list=stopwords
def sent(f):
    try:
        response = requests.post("https://hf.space/embed/KheireddineDaouadi/DzSenti/+/api/predict/", json={"data": [f]}).json()
         ###print(response)
        data = str(response["data"])
        ##print(data["label"])
        senti=data[12:20]
        score=float(data[74:80])
    except:
        senti="neutral"
        score="0.00"
    return (score,senti)

def topic(f):
    try:
        response = requests.post("https://hf.space/embed/KheireddineDaouadi/DzTopic/+/api/predict/", json={"data": [f]}).json()
         ###print(response)
        data = str(response["data"])
        ##print(data["label"])
        senti=data[12:20]
        score=float(data[74:80])
    except:
        senti="others"
        score="0.00"
    return (score,senti)
def hate(f):
    try:
        response = requests.post("https://hf.space/embed/KheireddineDaouadi/hate/+/api/predict/", json={"data": [f]}).json()
         ###print(response)
        data = str(response["data"])
        ##print(data["label"])
        senti=data[12:20]
        score=float(data[74:80])
    except:
        senti="others"
        score="0.00"
    return (score,senti)




def preprocess_ar(text):
    processedText = []
    
    # Create Lemmatizer and Stemmer.
    st = ISRIStemmer()
    commentwords = ''
    for t in text:
        t = ''.join(c for c in t if ud.category(c) == 'Lo' or ud.category(c) == 'Nd' or c == ' ')
        for word in t.split():
            # Checking if the word is a stopword.
            if word not in st_list:
                if len(word)>1:
                    # Lemmatizing the word.
                    word = st.suf32(word)
                    commentwords += (word+' ')
    processedText.append(commentwords)
    
    return processedText
def influ(l1):
    inf=[]
    linkinf=[]
    g=Counter(l1)
    df = pd.DataFrame.from_records(g.most_common(5), columns=['user'])
    return (df)
def wrdcld(text):
    processedtext_ar = preprocess_ar(text)
    awc = ArabicWordCloud(background_color="white")
    plt.figure(figsize = (16,16))
    wc_ar = awc.from_text(u''.join(processedtext_ar))
    image=plt.imshow(wc_ar)
    return (image)



headers = {"Authorization": f"Bearer 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"}

from datetime import datetime

pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', None)  # or 199s
from collections import Counter
import re
def clean_text(text):
    text = text.encode("ascii", errors="ignore").decode(
        "ascii"
    )  # remove non-ascii, Chinese characters
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\n\n", " ", text)
    text = re.sub(r"\t", " ", text)
    text = text.strip(" ")
    text = re.sub(
        " +", " ", text
    ).strip()  # get rid of multiple spaces and replace with a single
    return text
def clean_str(text):
    search = ["أ","إ","آ","ة","_","-","/",".","،"," و "," يا ",'"',"ـ","'","ى","\\",'\n', '\t','&quot;','?','؟','!',"يٰ"]
    replace = ["ا","ا","ا","ه"," "," ","","",""," و"," يا ","","","","ي","",' ', ' ',' ',' ? ',' ؟ ',' ! ','ي']
    p_tashkeel = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
    text = re.sub(p_tashkeel,"", text)
    p_longation = re.compile(r'(.)\1+')
    subst = r"\1\1"
    text = re.sub(p_longation, subst, text)
    text = text.replace('وو', 'و')
    text = text.replace('يي', 'ي')
    text = text.replace('اا', 'ا')
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("[ىيٰ]", "ي", text)
    text = text.replace('اا', 'ا')
    text = re.sub("ة", "ه", text)
    ##text = re.sub("گ", "ك", text)
    string1 = re.sub(r"http:\S+", '',str(text))
    string1 = re.sub(r"https:\S+", '',str(string1))
    string1 = re.sub(r"www.\S+", '',str(string1))
    string1 = re.sub(r"RT \S+", '',str(string1))
    string1 = re.sub(r"@\S+", '',str(string1))
    string1 = re.sub(r" RT ", '',str(string1))
    string1 = re.sub(r"\S+.com", '',str(string1))
    string1 = re.sub(r"#", " ",str(string1))
    string1 = re.sub(r"_", " ",str(string1))
    string1 = re.sub("\n", " ",str(string1))
    string1 = re.sub("\t", " ",str(string1))
    string1 = re.sub("\r", " ",str(string1))
    string1 = re.sub("۰", " ",str(string1))
    string1 = re.sub("۱", " ",str(string1))
    string1 = re.sub("۲", " ",str(string1))
    string1 = re.sub("۳", " ",str(string1))
    string1 = re.sub("۴", " ",str(string1))
    string1 = re.sub("۵", " ",str(string1))
    string1 = re.sub("۶", " ",str(string1))
    string1 = re.sub("۷", " ",str(string1))
    string1 = re.sub("۸", " ",str(string1))
    string1 = re.sub("۹", " ",str(string1))
    ##string1 = give_emoji_free_text(string1)
    string1 = re.sub(r'[^0-9\u0600-\u06ff\u0750-\u077f\ufb50-\ufbc1\ufbd3-\ufd3f\ufd50-\ufd8f\ufd50-\ufd8f\ufe70-\ufefc\uFDF0-\uFDFD.0-9]+', ' ', string1)
    text_tokens = word_tokenize(string1)
    string1 = ' '.join([word for word in text_tokens if not word in st_list])
    string1= ''.join([i for i in str(string1) if not i.isdigit()])
    string1 = re.sub('["*)#÷×%(،<>\*`؛`~≠&+@”_|…!,?.؟(“{}$_^;=:/-]', ' ', str(string1))
    string1 = re.sub(r"\s+", " ", str(string1))
    string1=' '.join( [w for w in string1.split() if len(w)>1] )
    string1 = string1.rstrip()
    text = string1.lstrip()
    for i in range(0, len(search)):
        text = text.replace(search[i], replace[i])   
    text = text.strip()
    return text
def get_comment (account_name):
    comments=[]
    reactors=[]
    treactors=[]
    nbrcomment=[]
    nbrshare=[]
    reaction=[[0 for i in range(10)] for i in range(10)]
    i=0
    ##url="https://www.facebook.com/TebbouneAbdelmadjid/posts/pfbid08XG2kL62AGuXanDjaVpv1zaAhuMjBFe7ivnMmKeSCeYiiRZgYAkHXtvUynJmeCpMlv"
    ##post_urls=[url]
    df2=pd.DataFrame(list(zip([],[])),columns=['name',"link"])
    df1=pd.DataFrame(list(zip([])),columns=['text'])
    print(df2)
    print(df1)
    sdd=0
    for post in get_posts("TebbouneAbdelmadjid", timeout=60, options={"comments" : True, "reactors": True,"allow_extra_requests": True,"progress": True},cookies="/kaggle/input/yuuuuuuuu/facebook.com_cookies.txt" ):
        if i<10:     
            df11=pd.DataFrame(list(zip(post["comments_full"][rd]["comment_text"] for rd in range(30))),columns=['text'])
            df1=pd.concat([df1,df11], ignore_index=True) 
            reaction[0][i]=post["reactions"]["like"] if 'like' in post["reactions"] else 0
            reaction[1][i]=post["reactions"]["love"] if 'love' in post["reactions"] else 0
            reaction[2][i]=post["reactions"]["care"] if 'care' in post["reactions"] else 0
            reaction[3][i]=post["reactions"]["haha"] if 'haha' in post["reactions"] else 0
            reaction[4][i]=post["reactions"]["wow"] if 'wow' in post["reactions"] else 0
            reaction[5][i]=post["reactions"]["angry"] if 'angry' in post["reactions"] else 0
            reaction[6][i]=post["reactions"]["sad"] if 'sad' in post["reactions"] else 0
            reaction[7][i]=post["time"].strftime("%m/%d/%h/%m/%s")
            reaction[8][i]=post["comments"]
            reaction[9][i]=post["shares"]       
            listreactname=(post["reactors"][gf]["link"] for gf in range(len(post["reactors"])))
            df2=pd.concat([df2,pd.DataFrame(list(zip(listreactname)),columns=["link"])], ignore_index=True)
            i=i+1
        else:
            break
    df1=df1["text"].astype('str').apply(clean_str)
    return (df1, df2, reaction)





def calculesent (comm):
    i=0
    allscore=[]
    allsent=[]
    for fg in comm:
        sco,sen = sent(fg)
        allscore.append(sco)
        allsent.append(sen)
    df=pd.DataFrame(list(zip(comm, allscore, allsent)),columns=['text','scoore1', 'label1'])
    df=df.sort_values(by = ['label1','scoore1'],ascending=False)
    print(len(df))
      ###df.to_csv('ot.csv')
    df1 = df[df['label1'] == 'positive' ]
    print(len(df1))
    df2 = df[df['label1'] == 'negative' ]
    return (df1,df2)

def visualise (reaction,text):
    data0 = {reaction[7][9]: reaction[0][9], reaction[7][8]: reaction[0][8], reaction[7][7]: reaction[0][7],reaction[7][6]: reaction[0][6],reaction[7][5]: reaction[0][5],reaction[7][4]: reaction[0][4],reaction[7][3]: reaction[0][3],reaction[7][2]: reaction[0][2],reaction[7][1]: reaction[0][1],reaction[7][0]: reaction[0][0]}
    data1 = {reaction[7][9]: reaction[1][9], reaction[7][8]: reaction[1][8], reaction[7][7]: reaction[1][7],reaction[7][6]: reaction[1][6],reaction[7][5]: reaction[1][5],reaction[7][4]: reaction[1][4],reaction[7][3]: reaction[1][3],reaction[7][2]: reaction[1][2],reaction[7][1]: reaction[1][1],reaction[7][0]: reaction[1][0]}
    data2 = {reaction[7][9]: reaction[2][9], reaction[7][8]: reaction[2][8], reaction[7][7]: reaction[2][7],reaction[7][6]: reaction[2][6],reaction[7][5]: reaction[2][5],reaction[7][4]: reaction[2][4],reaction[7][3]: reaction[2][3],reaction[7][2]: reaction[2][2],reaction[7][1]: reaction[2][1],reaction[7][0]: reaction[2][0]}
    data3 = {reaction[7][9]: reaction[3][9], reaction[7][8]: reaction[3][8], reaction[7][7]: reaction[3][7],reaction[7][6]: reaction[3][6],reaction[7][5]: reaction[3][5],reaction[7][4]: reaction[3][4],reaction[7][3]: reaction[3][3],reaction[7][2]: reaction[3][2],reaction[7][1]: reaction[3][1],reaction[7][0]: reaction[3][0]}
    data4 = {reaction[7][9]: reaction[4][9], reaction[7][8]: reaction[4][8], reaction[7][7]: reaction[4][7],reaction[7][6]: reaction[4][6],reaction[7][5]: reaction[4][5],reaction[7][4]: reaction[4][4],reaction[7][3]: reaction[4][3],reaction[7][2]: reaction[4][2],reaction[7][1]: reaction[4][1],reaction[7][0]: reaction[4][0]}
    data5 = {reaction[7][9]: reaction[5][9], reaction[7][8]: reaction[5][8], reaction[7][7]: reaction[5][7],reaction[7][6]: reaction[5][6],reaction[7][5]: reaction[5][5],reaction[7][4]: reaction[5][4],reaction[7][3]: reaction[5][3],reaction[7][2]: reaction[5][2],reaction[7][1]: reaction[5][1],reaction[7][0]: reaction[5][0]}
    data6 = {reaction[7][9]: reaction[6][9], reaction[7][8]: reaction[6][8], reaction[7][7]: reaction[6][7],reaction[7][6]: reaction[6][6],reaction[7][5]: reaction[6][5],reaction[7][4]: reaction[6][4],reaction[7][3]: reaction[6][3],reaction[7][2]: reaction[6][2],reaction[7][1]: reaction[6][1],reaction[7][0]: reaction[6][0]}
    data7 = {reaction[7][9]: reaction[8][9], reaction[7][8]: reaction[8][8], reaction[7][7]: reaction[8][7],reaction[7][6]: reaction[8][6],reaction[7][5]: reaction[8][5],reaction[7][4]: reaction[8][4],reaction[7][3]: reaction[8][3],reaction[7][2]: reaction[8][2],reaction[7][1]: reaction[8][1],reaction[7][0]: reaction[8][0]}
    data8 = { reaction[7][9]: reaction[9][9], reaction[7][8]: reaction[9][8], reaction[7][7]: reaction[9][7],reaction[7][6]: reaction[9][6],reaction[7][5]: reaction[9][5],reaction[7][4]: reaction[9][4],reaction[7][3]: reaction[9][3],reaction[7][2]: reaction[9][2],reaction[7][1]: reaction[9][1],reaction[7][0]: reaction[9][0]}
    print(data0)
    names0 = list(data0.keys())
    values0 = list(data0.values())
    names1 = list(data1.keys())
    values1 = list(data1.values())
    
    names2 = list(data2.keys())
    values2 = list(data2.values())
    names3 = list(data3.keys())
    values3 = list(data3.values())
    names4 = list(data4.keys())
    values4 = list(data4.values())
    names5 = list(data5.keys())
    values5 = list(data5.values())
    names6 = list(data6.keys())
    values6 = list(data6.values())

    names7 = list(data7.keys())
    values7 = list(data7.values())
    names8 = list(data8.keys())
    values8 = list(data8.values())
    
    fig, axs = plt.subplots(1, 1, figsize=(9,5), sharey=True)
    axs.plot(names0, values0,color='tab:blue')
    plt.xticks(rotation = 90)
    plt.title("Like Per Post")
    plt.ylabel("Number of Like")
    plt.xlabel("Post Date")
    
    fig1, axs1 = plt.subplots(1, 1, figsize=(9, 5), sharey=True)
    axs1.plot(names1, values1,color='aquamarine')
    plt.xticks(rotation = 90)
    plt.title("Love Per Post")
    plt.ylabel("Number of Love")
    plt.xlabel("Post Date")

    fig2, axs2 = plt.subplots(1, 1, figsize=(9, 5), sharey=True)
    axs2.plot(names2, values2,color='mediumseagreen')
    plt.xticks(rotation = 90)
    plt.title("Care Per Post")
    plt.ylabel("Number of Care")
    plt.xlabel("Post Date")

    fig1, axs1 = plt.subplots(1, 1, figsize=(9, 5), sharey=True)
    axs1.plot(names3, values3,color='xkcd:sky blue')
    plt.xticks(rotation = 90)
    plt.title("Haha Per Post")
    plt.ylabel("Number of Haha")
    plt.xlabel("Post Date")

    fig4, axs4 = plt.subplots(1, 1, figsize=(9, 5), sharey=True)
    axs4.plot(names4, values4,color='tab:purple')
    plt.xticks(rotation = 90)
    plt.title("Wow Per Post")
    plt.ylabel("Number of Wow")
    plt.xlabel("Post Date")

    fig5, axs5 = plt.subplots(1, 1, figsize=(5, 9), sharey=True)
    axs5.plot(names5, values5)

    fig6, axs6 = plt.subplots(1, 1, figsize=(5, 9), sharey=True)
    axs6.plot(names6, values6)

    fig7, axs7 = plt.subplots(1, 1, figsize=(5, 9), sharey=True)
    axs7.plot(names7, values7)
    fig8, axs8 = plt.subplots(1, 1, figsize=(5, 9), sharey=True)
    axs8.plot(names8, values8)
    
    
    text_ar= list(text)
    text_ar=preprocess_ar(text_ar)
    print(text_ar)
    text_ar=re.sub("\[","",str(text_ar))
    text_ar=re.sub("\]","",str(text_ar))
      ####print(str(text_ar).encode('unicode-escape').decode('utf-8'))
    data = arabic_reshaper.reshape(text_ar)
      ###print(data)
    data = get_display(data) # add this line
      ###print(data)
    WordCloud1 = WordCloud(font_path='arial', background_color='white',mode='RGB', width=2000, height=1000).generate(str(data))
##      im = Image.open(WordCloud1)
##      data = io.BytesIO()
##      im.save(data, "JPEG")
##      encoded_img_data = base64.b64encode(data.getvalue())
##    plt.imshow(WordCloud1)
##    plt.axis("off")
##    plt.show()
    return(fig1,fig2,fig3,fig4,fig5,fig6,fig7,fig8,WordCloud1)





def list2json (text,scor,sent):
    d = [ { 'text': x, 'score': y, 'sentiment': z } for x, y, z in zip(text, scor, sent) ]
    pretty_json = json.dumps(d, sort_keys=True, indent=4,ensure_ascii=False)
    return pretty_json
    

def query1(payload):
	response = requests.post(API_URL2, headers=headers, json=payload)
	return response.json()
def query(payload):
    data = json.dumps(payload)
    response = requests.request("POST", API_URL2, headers=headers, data=data)
    return json.loads(response.content.decode("utf-8"))









@app.route("/hello")
def index():
	flash("Please Select Function!!!!")
	return render_template("index.html")

@app.route("/greet", methods=['POST', 'GET'])
def greeter1():
        return render_template("sentimentoftext.html")
@app.route("/calculesent", methods=['POST', 'GET'])
def greeter2():
    if request.form['action'] == '😻 Get Sentiment ☹️':
                v,s=sent(request.form['name_input1'])
                c="The Sentiment Expressed In "+str(request.form['name_input1'])+"\nis :"+str(s)+"\nThe Sentiment score is : "+str(v)
                return render_template("sentimentoftext.html",v1=c)
    elif request.form['action'] == ' Topic Classification ':
                v,s=topic(request.form['name_input1'])
                c="The Topic Expressed In "+str(request.form['name_input1'])+"\nis :"+str(s)+"\nThe Topic scor is : "+str(v)
                return render_template("sentimentoftext.html",v1=c)
    elif request.form['action'] == ' Hateful Verification ':
                v,s=hate(request.form['name_input1'])
                c="Your Text "+str(request.form['name_input1'])+"\ncould be :"+str(s)+"\nThe hateful score is : "+str(v)
                return render_template("sentimentoftext.html",v1=c)
    else:
                return render_template("sentimentoftext.html")
        

@app.route("/retoursentiment", methods=['POST', 'GET'])
def greeter3():
	return render_template("index.html")

@app.route("/greet2", methods=['POST', 'GET'])
def greeter4():
	###flash("Hi " + str(request.form['name_input1']) + ", great to see you!")
	return render_template("login.html")

@app.route("/greet3", methods=['POST', 'GET'])
def greeter5():
	###flash("Hi " + str(request.form['name_input1']) + ", great to see you!")
	return render_template("login.html")


@app.route("/SBK", methods=['POST', 'GET'])
def greeter6():
	###flash("Hi " + str(request.form['name_input1']) + ", great to see you!")
	return render_template("SBK.html")

@app.route("/retourSBK", methods=['POST', 'GET'])
def greeter7():
	###flash("Hi " + str(request.form['name_input1']) + ", great to see you!")
	return render_template("index.html")


@app.route("/SFP", methods=['POST', 'GET'])

def greeter8():
    t = time.time()
    text,user,reaction=get_comment(account_name="TebbouneAbdelmadjid")
    tiiii=time.time()-t
    print(tiiii)
    t = time.time()
    text=list(text)[1:-2]
    print(text)
    dff1,dff2=calculesent(list(text))
    tiiii=time.time()-t
    print(tiiii)
    f1,f2,f2,f3,f4,f5,f6,f7,f8,F9= visualise(reaction,dff2)
    tiiii=time.time()-t
    plt.imshow(f1)
    f1.show()
    plt.imshow(f2)
    f2.show()
    plt.imshow(f3)
    f3.show()
    plt.imshow(f4)
    f4.show()
    plt.imshow(f5)
    f5.show()
    plt.imshow(f6)
    f6.show()
    plt.imshow(f7)
    f7.show()
    plt.imshow(f8)
    f8.show()
    plt.imshow(F9)
    F9.show()
    return render_template("SFP.html",v1=ff4,v2=ff5,v3=ff6,v4=ff1,v5=ff2,v6=ff3,wrd=encoded_img_data.decode('utf-8'))
@app.route("/retourSFP", methods=['POST', 'GET'])
def greeter9():
	###flash("Hi " + str(request.form['name_input1']) + ", great to see you!")
	return render_template("index.html")







    
@app.route('/pythonlogin/', methods=['GET', 'POST'])
def login():
    # Output message if something goes wrong...
    msg = ''
    # Check if "username" and "password" POST requests exist (user submitted form)
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        # Create variables for easy access
        username = request.form['username']
        password = request.form['password']
        # Check if account exists using MySQL
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE username = %s AND password = %s', (username, password,))
        # Fetch one record and return result
        account = cursor.fetchone()
        # If account exists in accounts table in out database
        if account:
            # Create session data, we can access this data in other routes
            session['loggedin'] = True
            session['id'] = account['id']
            session['username'] = account['username']
            # Redirect to home page
            return redirect(url_for('home'))
        else:
            # Account doesnt exist or username/password incorrect
            msg = 'Incorrect username/password!'
    # Show the login form with message (if any)
    return render_template('login.html', msg=msg)
# http://localhost:5000/python/logout - this will be the logout page
@app.route('/pythonlogin/logout')
def logout():
    # Remove session data, this will log the user out
   session.pop('loggedin', None)
   session.pop('id', None)
   session.pop('username', None)
   # Redirect to login page
   return redirect(url_for('login'))
# http://localhost:5000/pythinlogin/register - this will be the registration page, we need to use both GET and POST requests
@app.route('/pythonlogin/register', methods=['GET', 'POST'])
def register():
    # Output message if something goes wrong...
    msg = ''
    # Check if "username", "password" and "email" POST requests exist (user submitted form)
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'email' in request.form:
        # Create variables for easy access
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        # Check if account exists using MySQL
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE username = %s', (username,))
        account = cursor.fetchone()
        # If account exists show error and validation checks
        if account:
            msg = 'Account already exists!'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            msg = 'Invalid email address!'
        elif not re.match(r'[A-Za-z0-9]+', username):
            msg = 'Username must contain only characters and numbers!'
        elif not username or not password or not email:
            msg = 'Please fill out the form!'
        else:
            # Account doesnt exists and the form data is valid, now insert new account into accounts table
            cursor.execute('INSERT INTO accounts VALUES (NULL, %s, %s, %s)', (username, password, email,))
            mysql.connection.commit()
            msg = 'You have successfully registered!'
# http://localhost:5000/pythinlogin/home - this will be the home page, only accessible for loggedin users
@app.route('/pythonlogin/home')
def home():
    # Check if user is loggedin
    if 'loggedin' in session:
        # User is loggedin show them the home page
        return render_template('home.html', username=session['username'])
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))


# http://localhost:5000/pythinlogin/profile - this will be the profile page, only accessible for loggedin users
@app.route('/pythonlogin/profile')
def profile():
    # Check if user is loggedin
    if 'loggedin' in session:
        # We need all the account info for the user so we can display it on the profile page
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE id = %s', (session['id'],))
        account = cursor.fetchone()
        # Show the profile page with account info
        return render_template('profile.html', account=account)
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))
