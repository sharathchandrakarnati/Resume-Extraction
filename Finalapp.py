#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8   
# In[ ]:
    
    
import streamlit as st
from streamlit_option_menu import option_menu
st.set_option('deprecation.showPyplotGlobalUse', False)
import contractions 
import pandas as pd
import docx2txt
from PIL import Image 
from PyPDF2 import PdfFileReader
import pdfplumber
import numpy as np
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
import seaborn as sns
warnings.filterwarnings('ignore')
#for text pre-processing
import nltk
nltk.download('omw-1.4')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from wordcloud import WordCloud
from textblob import TextBlob, Word
nltk.download('stopwords')
stop=set(stopwords.words('english'))
from collections import  Counter
from nltk.util import ngrams
import re, unicodedata
import inflect
from nltk.stem import LancasterStemmer
nltk.download('omw-1.4')
from bs4 import BeautifulSoup
import spacy
import streamlit.components.v1 as components
import os
#for model-building
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
import xgboost as xgb
import string
from tqdm import tqdm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
#for model accuracy
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix, mean_absolute_error, mean_squared_error
from sklearn import preprocessing, decomposition, model_selection
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import absolute
from numpy import sqrt
# bag of words
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
#for visualization
import cufflinks as cf
import matplotlib
import matplotlib.pyplot as plt
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
from plotly import tools
import plotly.tools as tls
init_notebook_mode(connected=True)
import plotly.express as px
import plotly.graph_objects as go
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis.gensim_models
##pyLDAvis.enable_notebook()
import plotly.express as px
#for modelimplementation
import pickle
from joblib import dump, load
import joblib
# Utils
import time
timestr = time.strftime("%Y%m%d-%H%M%S")
import sys


###############################################Text Processing###############################################
#convert to lowercase, strip, remove punctuations, remove URL, remove HTML and remove emoji
def preprocess(text):
    text = text.lower() 
    text = text.strip()  
    text = re.compile('<.*?>').sub('', text) 
    text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)  
    text = re.sub('\s+', ' ', text)  
    text = re.sub(r'\[[0-9]*\]',' ',text) 
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
    text = re.sub(r'\d',' ',text) 
    text = re.sub(r'\s+',' ',text) 
    text = re.compile(r'https?://\S+|www\.\S+').sub(r'',text)
    text = re.compile(r'<.*?>').sub(r'',text)
    text = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE).sub(r'', text)
    return text

# STOPWORD REMOVAL
def stopword(string):
    a= [i for i in string.split() if i not in stopwords.words('english')]
    return ' '.join(a)

#LEMMATIZATION
# Initialize the lemmatizer
wl = WordNetLemmatizer()
 
# This is a helper function to map NLTK position tags
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


# Tokenize the sentence
def lemmatizer(string):
    word_pos_tags = nltk.pos_tag(word_tokenize(string)) # Get position tags
    a=[wl.lemmatize(tag[0], get_wordnet_pos(tag[1])) for idx, tag in enumerate(word_pos_tags)] # Map the position tag and lemmatize the word/token
    return " ".join(a)



def finalpreprocess(string):
    return lemmatizer(stopword(preprocess(string)))


###############################################Exploratory Data Analysis###############################################

#For Label Analysis
def label_analysis(df):
    #Final pre-processing
    df['clean_text'] = df['Extracted'].apply(lambda x: finalpreprocess(x))
    df['resume_len'] = df['Extracted'].astype(str).apply(len)
    df['Label_id']= df['Label'].factorize()[0]
    
    label2=df['Label'].value_counts().iplot(kind='bar',asFigure=True, bins=100,xTitle='Label',linecolor='black',yTitle='Number of Occurrences',title='Number Of Resume Under Each Label')
    st.plotly_chart(label2)
    st.header('Inference')

    st.markdown("""
           #### Resume Count Distribution over the Label:
           + 1. Peoplesoft resumes-           20
           + 2. SQL Developer resumes-        14
           + 3. Workday Resumes-              20
           + 4. React JS-                     22
           + 5. Internship-                   2               
           """)
    
    #Label_pie
    labels = list(df['Label'].value_counts().index)[0:]
    values = list(df['Label'].value_counts().values)[0:]
    colors = ['lightblue','gray','#eee','#999', '#9f9f']
    trace = go.Pie(labels=labels, values=values, hoverinfo='label+percent', 
                   textinfo='value', name='Resume counts of different category',
                   marker=dict(colors=colors))
    layout = dict(title = 'Pie Chart of Label',
                  xaxis= dict(title= 'Resume',ticklen= 5,zeroline= False)
                 )
    fig = dict(data = [trace], layout = layout)
    
    st.plotly_chart(fig)
    st.header('Inference')

    st.markdown("""
           #### Resume Label Count on Percentage (Pie Chart):
           + The pie chart shows percentage of data from the dataset under each category.               
           """)
    
    
    ###############Box Plot#####################
    #Final pre-processing
    df['clean_text'] = df['Extracted'].apply(lambda x: finalpreprocess(x))
    df['resume_len'] = df['Extracted'].astype(str).apply(len)
    
    y0 = df.loc[df['Label'] == 'Peoplesoft resumes']['resume_len']
    y1 = df.loc[df['Label'] == 'SQL Developer Lightning insight']['resume_len']
    y2 = df.loc[df['Label'] == 'workday resumes']['resume_len']
    y3 = df.loc[df['Label'] == 'React JS']['resume_len']
    y4 = df.loc[df['Label'] == 'Internship']['resume_len']

    trace0 = go.Box(
        y=y0,
        name = 'Peoplesoft resumes',
        marker = dict(
            color = 'rgb(214, 12, 140)',
        )
    )

    trace1 = go.Box(
        y=y1,
        name = 'SQL Developer Lightning insight',
        marker = dict(
            color = 'rgb(0, 128, 128)',
        )
    )

    trace2 = go.Box(
        y=y2,
        name = 'workday resumes',
        marker = dict(
            color = 'rgb(10, 140, 208)',
        )
    )

    trace3 = go.Box(
        y=y3,
        name = 'React JS',
        marker = dict(
            color = 'rgb(12, 102, 14)',
        )
    )

    trace4 = go.Box(
        y=y4,
        name = 'Internship',
        marker = dict(
            color = 'rgb(10, 0, 100)',
        )
    )

    data = [trace0, trace1, trace2, trace3, trace4]
    layout = go.Layout(
        title = "Resume Length Boxplot of Different Resume Category"
    )

    fig2 = go.Figure(data=data,layout=layout)
    #iplot(fig2, filename = "Resume Length Boxplot of Resume")
    st.plotly_chart(fig2, filename = "Resume Length Boxplot of Resume")
    st.header('Inference')

    st.markdown("""
           #### Resume length boxplot under different label resume:
           + The median resume length of React JS & Internship category are relative lower than those of the other resume category.
           """)

#For Character & Word Distribution
def char_word_dist(df):  
    character_count= df['Extracted'].str.len().iplot(kind='hist',bins=100,asFigure=True, xTitle='character count',linecolor='black',yTitle='count',title='Resume Text Character Count Distribution')
    st.plotly_chart(character_count)
    st.header('Inference')

    st.markdown("""
           #### Resume Character Count Distribution:
           + The histogram shows that resume range from 2k to 18k characters and generally it is between 2.5k to 8k characters.
           """)    
    
    word_count= df['Extracted'].str.split().map(lambda x: len(x)).iplot(kind='hist',bins=100, asFigure=True, xTitle='word count',linecolor='black',yTitle='count',title='Resume Text Word Count Distribution')
    
    st.plotly_chart(word_count)
    st.header('Inference')

    st.markdown("""
           #### Resume Word Count Distribution:
           + Data exploration at a word-level. It is clear that the number of words in resume ranges from 400 to 2500 and mostly falls between 500 to 700 words.
           """)

#N-Gram Analysis
#For uni-gram
def get_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

#For bi-gram
def get_top_n_bigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(2, 2)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

#For Tri-gram
def get_top_n_trigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(3, 3)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]


def ngram_ana(df):
     #After Cleaning
    #Final pre-processing
    df['clean_text'] = df['Extracted'].apply(lambda x: finalpreprocess(x))
    df['resume_len'] = df['Extracted'].astype(str).apply(len)
    df['Label_id']= df['Label'].factorize()[0]
    
    common_words = get_top_n_words(df['Extracted'], 10)
    for word, freq in common_words:
        print(word, freq)
    
    #Before cleaning    
    #Unigram Analysis 
    df1 = pd.DataFrame(common_words, columns = ['ResumeText' , 'count'])
    unigram1= df1.groupby('ResumeText').sum()['count'].sort_values(ascending=True).iplot(kind='bar', asFigure=True, yTitle='Count', linecolor='black',orientation='h', title='Top 10 words in resume before removing stop words')
    st.plotly_chart(unigram1)
    st.header('Inference')

    st.markdown("""
           #### Resume Top 10 words before removing stop words:
           + We can evidently see that stopwords such as “and", "the” and “in” dominate in resume.
           
           """)    
    
    #Bi-gram Analysis
    common_words = get_top_n_bigram(df['Extracted'], 10)
    for word, freq in common_words:
        print(word, freq)

    df2 = pd.DataFrame(common_words, columns = ['ResumeText' , 'count'])
    bigram1= df2.groupby('ResumeText').sum()['count'].sort_values(ascending=True).iplot(kind='bar', yTitle='Count', asFigure=True,  linecolor='black',orientation='h', title='Top 10 bigrams in resume before cleaning data')
    st.plotly_chart(bigram1)
    st.header('Inference')

    st.markdown("""
           #### Resume Top 10 words before cleaning data.  
           
           """)   
    
    #Tri-gram
    common_words = get_top_n_trigram(df['Extracted'], 10)
    for word, freq in common_words:
        print(word, freq)

    df3 = pd.DataFrame(common_words, columns = ['ResumeText' , 'count'])
    trigram1=df3.groupby('ResumeText').sum()['count'].sort_values(ascending=True).iplot(kind='bar', yTitle='Count', asFigure=True,  linecolor='black',orientation='h', title='Top 10 trigrams in resume Before cleaning data')
    st.plotly_chart(trigram1)
    st.header('Inference')

    st.markdown("""
           #### Resume Top Trigrams Words Frequency before cleaning data.
           
           """)  
    
    #After cleaning
    #Unigram
    common_words = get_top_n_words(df['clean_text'], 10)
    for word, freq in common_words:
        print(word, freq)

    df4 = pd.DataFrame(common_words, columns = ['ResumeText' , 'count'])
    unigram2=df4.groupby('ResumeText').sum()['count'].sort_values(ascending=True).iplot(kind='bar', yTitle='Count', asFigure=True,  linecolor='black',orientation='h', title='Top 10 words in resume after cleaning data')
    st.plotly_chart(unigram2)
    st.header('Inference')

    st.markdown("""
           #### Top 10 Words in Resume After cleaning data:
           + We can see our resume label words are now mostly occurring.
           
           """)  
    
    #Bigram
    common_words = get_top_n_bigram(df['clean_text'], 10)
    for word, freq in common_words:
        print(word, freq)

    df5 = pd.DataFrame(common_words, columns = ['ResumeText' , 'count'])
    bigram2=df5.groupby('ResumeText').sum()['count'].sort_values(ascending=True).iplot(kind='bar', yTitle='Count', asFigure=True,  linecolor='black',orientation='h', title='Top 10 bigrams in resume after cleaning data')
    st.plotly_chart(bigram2)
    st.header('Inference')

    st.markdown("""
           #### Top 10 Resume Bigrams Words after cleaning data.
           
           """)  
    
    #Trigram
    common_words = get_top_n_trigram(df['clean_text'], 10)
    for word, freq in common_words:
        print(word, freq)

    df6 = pd.DataFrame(common_words, columns = ['ResumeText' , 'count'])
    trigram2=df6.groupby('ResumeText').sum()['count'].sort_values(ascending=True).iplot(kind='bar', yTitle='Count', asFigure=True,  linecolor='black',orientation='h', title='Top 10 trigrams in resume after cleaning data')
    st.plotly_chart(trigram2)
    st.header('Inference')

    st.markdown("""
           #### Top 10 Resume Trigrams Words after cleaning data.
           
           """)

#Sentiment Analysis
def sentiment(df):
    #Final pre-processing
    df['clean_text'] = df['Extracted'].apply(lambda x: finalpreprocess(x))
    df['resume_len'] = df['Extracted'].astype(str).apply(len)
    df['Label_id']= df['Label'].factorize()[0]
    
    #Polarity shows the sentiment of a piece of text. It counts the negative and positive words and determines the polarity. 
    #The value ranges from -1 to 1 where -1 represents the negative sentiment, 
    #0 represents neutral and 1 represent positive sentiment.
    df['polarity'] = df['clean_text'].map(lambda text: TextBlob(text).sentiment.polarity)
    
    sentiment1=df['polarity'].iplot(kind='hist',bins=50,xTitle='polarity', asFigure=True, linecolor='black',yTitle='count',title='Sentiment Polarity Distribution')
    st.plotly_chart(sentiment1)
    st.header('Inference')

    st.markdown("""
           #### Resume Sentiment Analysis Distribution Plot:
           + Vast majority of the sentiment polarity scores are greater than zero, means most of them are pretty positive.
           
           """)
    
    ###############Sentiment Box Plot######################
    y0 = df.loc[df['Label'] == 'Peoplesoft resumes']['polarity']
    y1 = df.loc[df['Label'] == 'SQL Developer Lightning insight']['polarity']
    y2 = df.loc[df['Label'] == 'workday resumes']['polarity']
    y3 = df.loc[df['Label'] == 'React JS']['polarity']
    y4 = df.loc[df['Label'] == 'Internship']['polarity']

    trace0 = go.Box(
        y=y0,
        name = 'Peoplesoft resumes',
        marker = dict(
            color = 'rgb(214, 12, 140)',
        )
    )

    trace1 = go.Box(
        y=y1,
        name = 'SQL Developer Lightning insight',
        marker = dict(
            color = 'rgb(0, 128, 128)',
        )
    )

    trace2 = go.Box(
        y=y2,
        name = 'workday resumes',
        marker = dict(
            color = 'rgb(10, 140, 208)',
        )
    )

    trace3 = go.Box(
        y=y3,
        name = 'React JS',
        marker = dict(
            color = 'rgb(12, 102, 14)',
        )
    )

    trace4 = go.Box(
        y=y4,
        name = 'Internship',
        marker = dict(
            color = 'rgb(10, 0, 100)',
        )
    )

    data = [trace0, trace1, trace2, trace3, trace4]
    layout = go.Layout(
        title = "Sentiment Polarity Boxplot of Different Resume Category"
    )

    fig3 = go.Figure(data=data,layout=layout)
    #fig1=px.box(fig3, filename = "Sentiment Polarity Boxplot of Resume")
    st.plotly_chart(fig3, filename = "Sentiment Polarity Boxplot of Resume")
    st.header('Inference')

    st.markdown("""
           #### Sentiment boxplot under different label:
           + The highest sentiment polarity score was achieved by Internship resume category, rest all of the four category are between 0 to 0.2 mostly. The workday resume has the lowest median polarity score.
           """)  


#Part Of Speech & Top Common Words
def pos_common_words(df):
    #Final pre-processing
    df['clean_text'] = df['Extracted'].apply(lambda x: finalpreprocess(x))
    df['resume_len'] = df['Extracted'].astype(str).apply(len)
    
    blob = TextBlob(str(df['clean_text']))
    pos_df = pd.DataFrame(blob.tags, columns = ['word' , 'pos'])
    pos_df = pos_df.pos.value_counts()[:20]
    pos= pos_df.iplot(kind='bar',xTitle='POS',yTitle='count', asFigure=True, title='Top Part-of-speech tagging for resume corpus')
    
    st.plotly_chart(pos)
    st.header('Inference')

    st.markdown("""
           #### Top Part Of Speech:
           + Top part of speech is of Noun Singular, than Cardinal Numbers are there after that Adjective words are there.
           
           """)  
    
    #Common words
    corpus=[]
    new= df['Extracted'].str.split()
    new=new.values.tolist()
    corpus=[word for i in new for word in i]

    from collections import defaultdict
    dic=defaultdict(int)
    for word in corpus:
        if word in stop:
            dic[word]+=1
    
    counter=Counter(corpus)
    most=counter.most_common()
    plt.figure(figsize=(9,10))

    x, y= [], []
    for word,count in most[:40]:
        if (word not in stop):
            x.append(word)
            y.append(count)

    fig = plt.figure(figsize=(15, 8))        
    sns.barplot(x=y,y=x)
    st.pyplot(fig)
    st.header('Inference')

    st.markdown("""
           #### Resume Most Common Word Frequency:
           + From above we can see which words occur most frequently. If we do our word level analysis we can see PeopleSoft occur most frequently in the resume and than Workday, SQL and least ReactJS.
           """)      
    
#Wordcloud
def wordcloud_draw(dataset, color = 'white'):
            words = ' '.join(dataset)
            cleaned_word = ' '.join([word for word in words.split()
            if (word != 'news' and word != 'text')])
            wordcloud = WordCloud(stopwords = stop,
            background_color = color,
            width = 2500, height = 2500).generate(cleaned_word)
            plt.figure(1, figsize = (15,8))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            st.pyplot()
            
            
def wordcloud_resume(df):
    #Final pre-processing
    df['clean_text'] = df['Extracted'].apply(lambda x: finalpreprocess(x))
    df['resume_len'] = df['Extracted'].astype(str).apply(len)
    
    df['LabelId']= df['Label'].factorize()[0]

    Peoplesoft = df[df['LabelId'] == 0]
    Peoplesoft = Peoplesoft['Extracted']

    SQL_Developer = df[df['LabelId'] == 1]
    SQL_Developer = SQL_Developer['Extracted']

    Workday = df[df['LabelId'] == 2]
    Workday = Workday['Extracted']

    React_JS = df[df['LabelId'] == 3]
    React_JS = React_JS['Extracted']

    Internship = df[df['LabelId'] == 4]
    Internship = Internship['Extracted']
    
    print("Peoplesoft related words:")
    wordcloud_draw(Peoplesoft, 'white')
    st.header('Inference')

    st.markdown("""
           #### WordCloud on Peoplesoft Resume.
           
           """)    
    
    print("SQL_Developer related words:")
    wordcloud_draw(SQL_Developer, 'white')
    st.header('Inference')

    st.markdown("""
           #### WordCloud on SQL_Developer Resume.
           
           """)  

    print("Workday related words:")
    wordcloud_draw(Workday, 'white')
    st.header('Inference')

    st.markdown("""
           #### WordCloud on Workday Resume.
           
           """)  

    print("React_JS related words:")
    wordcloud_draw(React_JS, 'white')
    st.header('Inference')

    st.markdown("""
           #### WordCloud on React JS Resume.
           
           """)  

    print("Internship related words:")
    wordcloud_draw(Internship, 'white')
    st.header('Inference')

    st.markdown("""
           #### WordCloud on Internship Resume.
           
           """) 

    
#Named Entity Recognition (NER)

def plot_named_entity_barchart(text):
    nlp = spacy.load("en_core_web_sm")
    
    def _get_ner(text):
        doc=nlp(text)
        return [X.label_ for X in doc.ents]
    
    ent=text.apply(lambda x : _get_ner(x))
    ent=[x for sub in ent for x in sub]
    counter=Counter(ent)
    count=counter.most_common()
    
    x,y=map(list,zip(*count))

    # get unique list elements to pass to
    # streamlit app for dropdown options
    ent = set(ent)
    
    return ent, x, y
    
# visualise tokens per entity
def plot_most_common_named_entity_barchart(text, entity):
   
    nlp = spacy.load("en_core_web_sm")

    def _get_ner(text,ent):
        doc=nlp(text)
        return [X.text for X in doc.ents if X.label_ == ent]
    
    entity_filtered=text.apply(lambda x: _get_ner(x,entity))
    entity_filtered=[i for x in entity_filtered for i in x]
        
    counter=Counter(entity_filtered)
    x,y=map(list,zip(*counter.most_common(30)))
    sns.barplot(y,x).set_title(entity)       
        
    st.pyplot() 


def NER2(df):
    st.write("This dashboard firstly offers NER instance count in input.")
    #Final pre-processing
    df['clean_text'] = df['Extracted'].apply(lambda x: finalpreprocess(x))
    df['resume_len'] = df['Extracted'].astype(str).apply(len)
    ent, x, y = plot_named_entity_barchart(df['clean_text'])
        
    sns.barplot(x=y,y=x)
    plt.xlabel('Row number')
    plt.ylabel('NER')
            
    plt.title('NER instance count\n\n',fontweight ="bold")

    # show plot in streamlit
    st.pyplot()
        
    st.write("Secondly, it offers specific entity count in input.")
        
    selected_entity = st.multiselect("Choose entity to quantify", (list(ent)))

    if selected_entity:
        plot_most_common_named_entity_barchart(df['clean_text'], entity = selected_entity[0])
        
    
    
#Topic Model Analysis
def preprocess_news(df):
    p = open("lda.html")
    st.components.v1.html(p.read(), width=1300, height=800, scrolling=True)
    st.header('Inference')

    st.markdown("""
           #### Topic modeling is the process of using unsupervised learning techniques to extract the main topics that occur in a collection of documents. We have used PyLDAvis.:
           + On the left side, the area of each circle represents the importance of the topic relative to the corpus. As there are five topics, we have five circles.
           + The distance between the center of the circles indicates the similarity between the topics. Here you can see that the topic 5,4,3('React JS, Internship, SQL server’) are near to each other, this indicates that the topics are more similar.
           + On the right side, the histogram of each topic shows the top 30 relevant words. For example, in topic 1('Workday') the most relevant words are Workday, integration, HCM, business, etc
           + So in our case, we can see a lot of words and topics associated with PeopleSoft and Workday.
           """) 

############################################### Model Learning ###############################################

#multi-class log-loss
def multiclass_logloss(actual, predicted, eps=1e-15):
    """Multi class version of Logarithmic Loss metric.
    :param actual: Array containing the actual target classes
    :param predicted: Matrix with class predictions, one probability per class
    """
    # Convert 'actual' to a binary array if it's not already:
    if len(actual.shape) == 1:
        actual2 = np.zeros((actual.shape[0], predicted.shape[1]))
        for i, val in enumerate(actual):
            actual2[i, val] = 1
        actual = actual2

    clip = np.clip(predicted, eps, 1 - eps)
    rows = actual.shape[0]
    vsota = np.sum(actual * np.log(clip))
    return -1.0 / rows * vsota

# CountVectorizer
def cv(data):
    count_vectorizer = CountVectorizer(analyzer='word',token_pattern=r'\w{1,}',ngram_range=(1, 3), stop_words = 'english')
    emb = count_vectorizer.fit_transform(data)
    return emb, count_vectorizer

#Term Frequency-Inverse Document Frequencies (tf-Idf)
def tfidf(data):
    tfidf_vectorizer = TfidfVectorizer(min_df=3,  max_features=None,strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',      ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1,stop_words = 'english')
    train = tfidf_vectorizer.fit_transform(data)
    return train, tfidf_vectorizer

#Model Logistic Regression
def logistic_regression(X_train, y_train, X_test, y_test):
    #cv2 = LeaveOneOut()
    lr= LogisticRegression(solver = 'liblinear', C=10, penalty = 'l2')
    lr.fit(X_train, y_train)
    
    #predict y value for dataset
    y_predict= lr.predict(X_test)
    y_prob= lr.predict_proba(X_test)
    
    report= classification_report(y_test,y_predict, output_dict=True)
    tab=pd.DataFrame(report).transpose()
    st.write(tab)
   
    
    logloss=multiclass_logloss(y_test, y_prob)
    st.write("logloss: %0.3f " % logloss) 
    
    #use LOOCV to evaluate model
    #scores= cross_val_score(lr, X_train, y_train, scoring='neg_mean_absolute_error',cv=cv2, n_jobs=-1)
    #view mean absolute error
    #MAE=mean(absolute(scores))
    #RMSE=sqrt(mean(absolute(scores)))
    #st.write("Mean Absolute Error: ", MAE.round(2))
    #st.write("Root Mean Squared Error: ", RMSE.round(2))
    
    st.subheader("Confusion Matrix") 
    conf_matrix= confusion_matrix(y_test, y_predict)
    sns.heatmap(conf_matrix, annot=True, cmap='Blues')
    st.pyplot()

# Logistic Regression CountVectorizer
def logistic_reg_count(df):
    #Final pre-processing
    df['clean_text'] = df['Extracted'].apply(lambda x: finalpreprocess(x))
    df['resume_len'] = df['Extracted'].astype(str).apply(len)
    df['LabelId']= df['Label'].factorize()[0]
    
    #SPLITTING THE TRAINING DATASET INTO TRAIN AND TEST
    X_train, X_test, y_train, y_test = train_test_split(df["clean_text"],df["LabelId"],test_size=0.3, random_state=30,shuffle=True)
    
    #CountVectorizer
    X_train_counts, count_vectorizer = cv(X_train)
    X_test_counts = count_vectorizer.transform(X_test)
    
    #Result
    logistic_regression(X_train_counts, y_train, X_test_counts , y_test)

#Logistic Regression TF-IDF    
def logistic_reg_tfiidf(df):
    #Final pre-processing
    df['clean_text'] = df['Extracted'].apply(lambda x: finalpreprocess(x))
    df['resume_len'] = df['Extracted'].astype(str).apply(len)
    df['LabelId']= df['Label'].factorize()[0]
    
    #SPLITTING THE TRAINING DATASET INTO TRAIN AND TEST
    X_train, X_test, y_train, y_test = train_test_split(df["clean_text"],df["LabelId"],test_size=0.3, random_state=30,shuffle=True)
    
    #Tf-Idf
    X_train_vectors_tfidf, tfidf_vectorizer = tfidf(X_train) 
    X_test_vectors_tfidf = tfidf_vectorizer.transform(X_test)
    
    #Result
    logistic_regression(X_train_vectors_tfidf, y_train, X_test_vectors_tfidf , y_test)

    
#Model XGB Classifier
def xgb_classifier(X_train, y_train, X_test, y_test):
    clf = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8, 
                        subsample=0.8, nthread=10, learning_rate=0.1)
    clf.fit(X_train.tocsc(), y_train)
    y_predict = clf.predict(X_test.tocsc())
    y_prob= clf.predict_proba(X_test.tocsc())
    
    report=classification_report(y_test,y_predict, output_dict=True)
    tab2=pd.DataFrame(report).transpose()
    st.write(tab2)
    
    logloss=multiclass_logloss(y_test, y_prob)
    st.write("logloss: %0.3f " % logloss) 
    
    #use LOOCV to evaluate model
    #cv3 = LeaveOneOut()
    #scores= cross_val_score(clf, X_train.tocsc(), y_train, scoring='neg_mean_absolute_error',cv=cv3, n_jobs=-1)
    #view mean absolute error
    #MAE=mean(absolute(scores))
    #RMSE=sqrt(mean(absolute(scores)))
    #st.write("Mean Absolute Error: ", MAE.round(2))
    #st.write("Root Mean Squared Error: ", RMSE.round(2))
    
    st.subheader("Confusion Matrix") 
    conf_matrix= confusion_matrix(y_test, y_predict)
    sns.heatmap(conf_matrix, annot=True, cmap='Blues')
    st.pyplot()
    
    
#XGB Classifier Countvectorizer
def xgb_count(df):
    #Final pre-processing
    df['clean_text'] = df['Extracted'].apply(lambda x: finalpreprocess(x))
    df['resume_len'] = df['Extracted'].astype(str).apply(len)
    df['LabelId']= df['Label'].factorize()[0]
    
    #SPLITTING THE TRAINING DATASET INTO TRAIN AND TEST
    X_train, X_test, y_train, y_test = train_test_split(df["clean_text"],df["LabelId"],test_size=0.3, random_state=30,shuffle=True)
    
    #CountVectorizer
    X_train_counts, count_vectorizer = cv(X_train)
    X_test_counts = count_vectorizer.transform(X_test)
    
    #Result
    xgb_classifier(X_train_counts, y_train, X_test_counts , y_test)
    
#XGB Classifier TFIDF
def xgb_tfidf(df):
    #Final pre-processing
    df['clean_text'] = df['Extracted'].apply(lambda x: finalpreprocess(x))
    df['resume_len'] = df['Extracted'].astype(str).apply(len)
    df['LabelId']= df['Label'].factorize()[0]
    
    #SPLITTING THE TRAINING DATASET INTO TRAIN AND TEST
    X_train, X_test, y_train, y_test = train_test_split(df["clean_text"],df["LabelId"],test_size=0.3, random_state=30,shuffle=True)
    
    #Tf-Idf
    X_train_vectors_tfidf, tfidf_vectorizer = tfidf(X_train) 
    X_test_vectors_tfidf = tfidf_vectorizer.transform(X_test)
    
    #Result
    xgb_classifier(X_train_vectors_tfidf, y_train, X_test_vectors_tfidf , y_test)
    
#Model SVM Classifier
def svm_classifier(X_train, y_train, X_test, y_test):
    svd = decomposition.TruncatedSVD(n_components=120)
    svd.fit(X_train)
    xtrain_svd = svd.transform(X_train)
    xtest_svd = svd.transform(X_test)

    # Scale the data obtained from SVD. Renaming variable to reuse without scaling.
    scl = preprocessing.StandardScaler(with_mean=False)
    scl.fit(xtrain_svd)
    xtrain_svd_scl = scl.transform(xtrain_svd)
    xtest_svd_scl = scl.transform(xtest_svd)
    
    # Fitting a simple SVM
    clf = SVC(C=1.0, probability=True) # since we need probabilities
    clf.fit(xtrain_svd_scl, y_train)
    y_predict = clf.predict(xtest_svd_scl)
    y_prob= clf.predict_proba(xtest_svd_scl)
    
    report=classification_report(y_test,y_predict, output_dict=True)
    tab3=pd.DataFrame(report).transpose()
    st.write(tab3)
    
    logloss=multiclass_logloss(y_test, y_prob)
    st.write("logloss: %0.3f " % logloss)  
    
    #use LOOCV to evaluate model
    #cv4 = LeaveOneOut()
    #scores= cross_val_score(clf, xtrain_svd_scl, y_train, scoring='neg_mean_absolute_error',cv=cv4, n_jobs=-1)
    #view mean absolute error
    #MAE=mean(absolute(scores))
    #RMSE=sqrt(mean(absolute(scores)))
    #st.write("Mean Absolute Error: ", MAE.round(2))
    #st.write("Root Mean Squared Error: ", RMSE.round(2))
    
    st.subheader("Confusion Matrix") 
    conf_matrix= confusion_matrix(y_test, y_predict)
    sns.heatmap(conf_matrix, annot=True, cmap='Blues')
    st.pyplot()

#SVM Classifier CountVectorizer
def svm_count(df):
    #Final pre-processing
    df['clean_text'] = df['Extracted'].apply(lambda x: finalpreprocess(x))
    df['resume_len'] = df['Extracted'].astype(str).apply(len)
    df['LabelId']= df['Label'].factorize()[0]
    
    #SPLITTING THE TRAINING DATASET INTO TRAIN AND TEST
    X_train, X_test, y_train, y_test = train_test_split(df["clean_text"],df["LabelId"],test_size=0.3, random_state=30,shuffle=True)
    
    #CountVectorizer
    X_train_counts, count_vectorizer = cv(X_train)
    X_test_counts = count_vectorizer.transform(X_test)
    
    #Result
    svm_classifier(X_train_counts, y_train, X_test_counts , y_test)

#SVM Classifier TFIDF
def svm_tfidf(df):
    #Final pre-processing
    df['clean_text'] = df['Extracted'].apply(lambda x: finalpreprocess(x))
    df['resume_len'] = df['Extracted'].astype(str).apply(len)
    df['LabelId']= df['Label'].factorize()[0]
    
    #SPLITTING THE TRAINING DATASET INTO TRAIN AND TEST
    X_train, X_test, y_train, y_test = train_test_split(df["clean_text"],df["LabelId"],test_size=0.3, random_state=30,shuffle=True)
    
    #Tf-Idf
    X_train_vectors_tfidf, tfidf_vectorizer = tfidf(X_train) 
    X_test_vectors_tfidf = tfidf_vectorizer.transform(X_test)
    
    #Result
    svm_classifier(X_train_vectors_tfidf, y_train, X_test_vectors_tfidf , y_test)
    
#Model Decision Tree Classifier

def decisiontree_classifier(X_train, y_train, X_test, y_test):
    clf= DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    
    #predict y value for dataset
    y_predict= clf.predict(X_test)
    y_prob= clf.predict_proba(X_test)
    
    report=classification_report(y_test,y_predict, output_dict=True)
    tab4=pd.DataFrame(report).transpose()
    st.write(tab4)
    
    logloss=multiclass_logloss(y_test, y_prob)
    st.write("logloss: %0.3f " % logloss)  
    
    #use LOOCV to evaluate model
    #cv5 = LeaveOneOut()
    #scores= cross_val_score(clf, X_train, y_train, scoring='neg_mean_absolute_error',cv=cv5, n_jobs=-1)
    #view mean absolute error
    #MAE=mean(absolute(scores))
    #RMSE=sqrt(mean(absolute(scores)))
    #st.write("Mean Absolute Error: ", MAE.round(2))
    #st.write("Root Mean Squared Error: ", RMSE.round(2))
    
    st.subheader("Confusion Matrix")
    conf_matrix= confusion_matrix(y_test, y_predict)
    sns.heatmap(conf_matrix, annot=True, cmap='Blues')
    st.pyplot()
    
#Decision Tree Classifier CountVectorizer
def dec_tree_count(df):
    #Final pre-processing
    df['clean_text'] = df['Extracted'].apply(lambda x: finalpreprocess(x))
    df['resume_len'] = df['Extracted'].astype(str).apply(len)
    df['LabelId']= df['Label'].factorize()[0]
    
    #SPLITTING THE TRAINING DATASET INTO TRAIN AND TEST
    X_train, X_test, y_train, y_test = train_test_split(df["clean_text"],df["LabelId"],test_size=0.3, random_state=30,shuffle=True)
    
    #CountVectorizer
    X_train_counts, count_vectorizer = cv(X_train)
    X_test_counts = count_vectorizer.transform(X_test)
    
    #Result
    decisiontree_classifier(X_train_counts, y_train, X_test_counts , y_test)

    
#Decision Tree Classifier TFIDF    
def dec_tree_tfidf(df):
    #Final pre-processing
    df['clean_text'] = df['Extracted'].apply(lambda x: finalpreprocess(x))
    df['resume_len'] = df['Extracted'].astype(str).apply(len)
    df['LabelId']= df['Label'].factorize()[0]
    
    #SPLITTING THE TRAINING DATASET INTO TRAIN AND TEST
    X_train, X_test, y_train, y_test = train_test_split(df["clean_text"],df["LabelId"],test_size=0.3, random_state=30,shuffle=True)
    
    #Tf-Idf
    X_train_vectors_tfidf, tfidf_vectorizer = tfidf(X_train) 
    X_test_vectors_tfidf = tfidf_vectorizer.transform(X_test)
    
    #Result
    decisiontree_classifier(X_train_vectors_tfidf, y_train, X_test_vectors_tfidf , y_test)
    
    
#Model Random Forest Classifier
def randomforest_classifier(X_train, y_train, X_test, y_test):
    clf= RandomForestClassifier()
    clf.fit(X_train, y_train)
    
    #predict y value for dataset
    y_predict= clf.predict(X_test)
    y_prob= clf.predict_proba(X_test)
    
    report=classification_report(y_test,y_predict, output_dict=True)
    tab5=pd.DataFrame(report).transpose()
    st.write(tab5)
    
    logloss=multiclass_logloss(y_test, y_prob)
    st.write("logloss: %0.3f " % logloss)
    
    #use LOOCV to evaluate model
    #cv6 = LeaveOneOut()
    #scores= cross_val_score(clf, X_train, y_train, scoring='neg_mean_absolute_error',cv=cv6, n_jobs=-1)
    #view mean absolute error
    #MAE=mean(absolute(scores))
    #RMSE=sqrt(mean(absolute(scores)))
    #st.write("Mean Absolute Error: ", MAE.round(2))
    #st.write("Root Mean Squared Error: ", RMSE.round(2))
        
    st.subheader("Confusion Matrix")    
    conf_matrix= confusion_matrix(y_test, y_predict)
    sns.heatmap(conf_matrix, annot=True, cmap='Blues')
    st.pyplot()
    
#Random Forest Classifier CountVectorizer
def random_forest_count(df):
    #Final pre-processing
    df['clean_text'] = df['Extracted'].apply(lambda x: finalpreprocess(x))
    df['resume_len'] = df['Extracted'].astype(str).apply(len)
    df['LabelId']= df['Label'].factorize()[0]
    
    #SPLITTING THE TRAINING DATASET INTO TRAIN AND TEST
    X_train, X_test, y_train, y_test = train_test_split(df["clean_text"],df["LabelId"],test_size=0.3, random_state=30,shuffle=True)
    
    #CountVectorizer
    X_train_counts, count_vectorizer = cv(X_train)
    X_test_counts = count_vectorizer.transform(X_test)
    
    #Result
    randomforest_classifier(X_train_counts, y_train, X_test_counts , y_test)
    
#Random Forest Classifier TFIDF
def random_forest_tfidf(df):
    #Final pre-processing
    df['clean_text'] = df['Extracted'].apply(lambda x: finalpreprocess(x))
    df['resume_len'] = df['Extracted'].astype(str).apply(len)
    df['LabelId']= df['Label'].factorize()[0]
    
    #SPLITTING THE TRAINING DATASET INTO TRAIN AND TEST
    X_train, X_test, y_train, y_test = train_test_split(df["clean_text"],df["LabelId"],test_size=0.3, random_state=30,shuffle=True)
    
    #Tf-Idf
    X_train_vectors_tfidf, tfidf_vectorizer = tfidf(X_train) 
    X_test_vectors_tfidf = tfidf_vectorizer.transform(X_test)
    
    #Result
    randomforest_classifier(X_train_vectors_tfidf, y_train, X_test_vectors_tfidf , y_test)


#Model KNN Classifier     
def kneighbors_classifier(X_train, y_train, X_test, y_test):
    clf= KNeighborsClassifier()
    clf.fit(X_train, y_train)
    
    #predict y value for dataset
    y_predict= clf.predict(X_test)
    y_prob= clf.predict_proba(X_test)
    
    report=classification_report(y_test,y_predict, output_dict=True)
    tab6=pd.DataFrame(report).transpose()
    st.write(tab6)
    
    logloss=multiclass_logloss(y_test, y_prob)
    st.write("logloss: %0.3f " % logloss)  
    
    #use LOOCV to evaluate model
    #cv7 = LeaveOneOut()
    #scores= cross_val_score(clf, X_train, y_train, scoring='neg_mean_absolute_error',cv=cv7, n_jobs=-1)
    #view mean absolute error
    #MAE=mean(absolute(scores))
    #RMSE=sqrt(mean(absolute(scores)))
    #st.write("Mean Absolute Error: ", MAE.round(2))
    #st.write("Root Mean Squared Error: ", RMSE.round(2))
        
    st.subheader("Confusion Matrix")     
    conf_matrix= confusion_matrix(y_test, y_predict)
    sns.heatmap(conf_matrix, annot=True, cmap='Blues')
    st.pyplot()
    
#KNN Classifier CountVectorizer
def knn_count(df):
    #Final pre-processing
    df['clean_text'] = df['Extracted'].apply(lambda x: finalpreprocess(x))
    df['resume_len'] = df['Extracted'].astype(str).apply(len)
    df['LabelId']= df['Label'].factorize()[0]
    
    #SPLITTING THE TRAINING DATASET INTO TRAIN AND TEST
    X_train, X_test, y_train, y_test = train_test_split(df["clean_text"],df["LabelId"],test_size=0.3, random_state=30,shuffle=True)
    
    #CountVectorizer
    X_train_counts, count_vectorizer = cv(X_train)
    X_test_counts = count_vectorizer.transform(X_test)
    
    #Result
    kneighbors_classifier(X_train_counts, y_train, X_test_counts , y_test)
    
#KNN Classifier TFIDF
def knn_tfidf(df):
    #Final pre-processing
    df['clean_text'] = df['Extracted'].apply(lambda x: finalpreprocess(x))
    df['resume_len'] = df['Extracted'].astype(str).apply(len)
    df['LabelId']= df['Label'].factorize()[0]
    
    #SPLITTING THE TRAINING DATASET INTO TRAIN AND TEST
    X_train, X_test, y_train, y_test = train_test_split(df["clean_text"],df["LabelId"],test_size=0.3, random_state=30,shuffle=True)
    
    #Tf-Idf
    X_train_vectors_tfidf, tfidf_vectorizer = tfidf(X_train) 
    X_test_vectors_tfidf = tfidf_vectorizer.transform(X_test)
    
    #Result
    kneighbors_classifier(X_train_vectors_tfidf, y_train, X_test_vectors_tfidf , y_test)
    
#Model Naive Bayes Classifier
def naive_bayes(X_train, y_train, X_test, y_test):
    # Fitting a simple Naive Bayes
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    
    y_predict = clf.predict(X_test)
    y_prob= clf.predict_proba(X_test)
    
    report=classification_report(y_test,y_predict, output_dict=True)
    tab7=pd.DataFrame(report).transpose()
    st.write(tab7)
    
    logloss=multiclass_logloss(y_test, y_prob)
    st.write("logloss: %0.3f " % logloss)
    
    #use LOOCV to evaluate model
    #cv8 = LeaveOneOut()
    #scores= cross_val_score(clf, X_train, y_train, scoring='neg_mean_absolute_error',cv=cv8, n_jobs=-1)
    #view mean absolute error
    #MAE=mean(absolute(scores))
    #RMSE=sqrt(mean(absolute(scores)))
    #st.write("Mean Absolute Error: ", MAE.round(2))
    #st.write("Root Mean Squared Error: ", RMSE.round(2))
        
    st.subheader("Confusion Matrix")     
    conf_matrix= confusion_matrix(y_test, y_predict)
    sns.heatmap(conf_matrix, annot=True, cmap='Blues')
    st.pyplot()
    
#Naive Bayes Classifier CountVectorizer
def naive_count(df):
    #Final pre-processing
    df['clean_text'] = df['Extracted'].apply(lambda x: finalpreprocess(x))
    df['resume_len'] = df['Extracted'].astype(str).apply(len)
    df['LabelId']= df['Label'].factorize()[0]
    
    #SPLITTING THE TRAINING DATASET INTO TRAIN AND TEST
    X_train, X_test, y_train, y_test = train_test_split(df["clean_text"],df["LabelId"],test_size=0.3, random_state=30,shuffle=True)
    
    #CountVectorizer
    X_train_counts, count_vectorizer = cv(X_train)
    X_test_counts = count_vectorizer.transform(X_test)
    
    #Result
    naive_bayes(X_train_counts, y_train, X_test_counts , y_test)
    
#Naive Bayes Classifier TFIDF
def naive_tfidf(df):
     #Final pre-processing
    df['clean_text'] = df['Extracted'].apply(lambda x: finalpreprocess(x))
    df['resume_len'] = df['Extracted'].astype(str).apply(len)
    df['LabelId']= df['Label'].factorize()[0]
    
    #SPLITTING THE TRAINING DATASET INTO TRAIN AND TEST
    X_train, X_test, y_train, y_test = train_test_split(df["clean_text"],df["LabelId"],test_size=0.3, random_state=30,shuffle=True)
    
    #Tf-Idf
    X_train_vectors_tfidf, tfidf_vectorizer = tfidf(X_train) 
    X_test_vectors_tfidf = tfidf_vectorizer.transform(X_test)
    
    #Result
    naive_bayes(X_train_vectors_tfidf, y_train, X_test_vectors_tfidf , y_test)

############################################### Prediction ###############################################

#For Dataset Classification
loaded_vec2 = pickle.load(open("tfidf_vect_dataset.pkl", "rb"))
res= pickle.load(open('textclassification.pkl','rb'))
#For Text Classification
clff = pickle.load(open('clf_resume.pkl','rb'))
loaded_vec = pickle.load(open("tfidf_vect.pkl", "rb"))
            
def read_pdf(file):
        pdfReader = PdfFileReader(file)
        count = pdfReader.numPages
        all_page_text = ""
        for i in range(count):
            page = pdfReader.getPage(i)
            all_page_text += page.extractText()

        return all_page_text

def read_pdf_with_pdfplumber(file):
        with pdfplumber.open(file) as pdf:
            page = pdf.pages[0]
            return page.extract_text()   
        
def remove_html(words):
    # language agnostic
    soup = BeautifulSoup(words, 'lxml')
    clean_words = soup.get_text()
    return clean_words

def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = [unicodedata.normalize('NFKD', w).encode('ascii', 'ignore').decode('utf-8', 'ignore') for w in words]
    return new_words

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = [word.lower() for word in words]
    return new_words

def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = [re.sub(r'[^\w\s]', '', word) for word in words]

    return new_words

def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words

def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words

def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems

def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()

    lemmas = [lemmatizer.lemmatize(i, pos="a") for i in words]

    lemmas = " ".join(lemmas)

    return lemmas

def normalize(words):
    #de-noising objects
    url_pattern = re.compile(re.compile(r'https?://\S+|www\.S+'))
    email_pattern = re.compile(re.compile(r'\S*@\S*\s?'))

    words = [url_pattern.sub('', w) for w in words]
    words = [email_pattern.sub('', w) for w in words]
    
    words = remove_non_ascii(words)
    
    words = to_lowercase(words)
    
    words = remove_punctuation(words)

    words = replace_numbers(words)
   
    words = remove_stopwords(words)

    return words

def text_resume(input3):
    result_pred = clff.predict(loaded_vec.transform(input3))
    st.success('The predicted LabelId is {}'.format(result_pred))
    st.markdown("""
           #### Data Classification on Resume, LabelId Resemble Below Resume:
           + 1. Peoplesoft resumes-           [0]
           + 2. SQL Developer resumes-        [1]
           + 3. Workday Resumes-              [2]
           + 4. React JS-                     [3]
           + 5. Internship-                   [4]               
           """)                            
                                                                         
    if result_pred == 0:
        image= Image.open("PeopleSoft_logo.svg.png")
        st.image(image,use_column_width=True)
    elif result_pred == 1:
        image= Image.open("sql Developer.png")
        st.image(image,use_column_width=True)    
    elif result_pred == 2:
        image= Image.open("workday.png")
        st.image(image,use_column_width=True)    
    elif result_pred == 3:
        image= Image.open("ReactJS.png")
        st.image(image,use_column_width=True)    
    elif result_pred == 4:
        image= Image.open("internship.png")
        st.image(image,use_column_width=True)                               

###############################################Streamlit Main###############################################

def main():
    # set page title
    st.set_page_config('Document Classifier')
    
            
    # 2. horizontal menu with custom style
    selected = option_menu(menu_title=None, options=["Home", "Projects", "About"], icons=["house", "book", "envelope"],  menu_icon="cast", default_index=0, orientation="horizontal", styles={"container": {"padding": "0!important", "background-color": "#fafafa"},"icon": {"color": "orange", "font-size": "25px"}, "nav-link": {"font-size": "25px","text-align": "left","margin": "0px","--hover-color": "#eee", },           "nav-link-selected": {"background-color": "green"},},)
    
    #horizontal Home selected
    if selected == "Home":
        st.title(f"You have selected {selected}")
        
        image= Image.open("nlp_streamlit.jpg")
        st.image(image,use_column_width=True)
            
        st.sidebar.title("Home")        
        with st.sidebar:
            image= Image.open("Home.png")
            add_image=st.image(image,use_column_width=True)  
        st.sidebar.markdown("[ Visit To Github Repositories](https://github.com/RahulSingh409/Project-Document-Classification.git)")    
        
        st.title('Data Classification')
        st.video("https://youtu.be/O73OPzkUlR0")
        
        st.markdown("""

             This is a Natural Language Processing(NLP) Based App useful for data classification.
            """)
        ### features

        st.header('Features')

        st.markdown("""
                #### Basic NLP Tasks:
                + App covers the most basic NLP task of Label Analysis, Character and Word Analysis, N-Gram Analysis, Correlation between variable, tokenisation, parts of speech tagging.
                
                #### Named Entity Recognition, Sentimental Analysis, WordCloud and Topic Modelling:
                + Named Entites like organistion person etc are recognised, Wordcloud as per label, Analysis on Sentiment and top topics from the corpus are found based on LDA modelling.
                #### Machine Learning:
                + Machine Learning on different Machne Algorithms, modeling with different classifier and vectorization technique lastly  prediction. 
                
                """)
                
    #Horizontal About selected
    if selected == "About":
        st.title(f"You have selected {selected}")
        
        st.sidebar.title("About")
        with st.sidebar:
            image= Image.open("About_us.png")
            add_image=st.image(image,use_column_width=True)        
        
        st.image('iidt_logo_137.png',use_column_width=True)
        st.markdown("<h2 style='text-align: center;'> This Project completed under ExcelR, the team completed the project:</h2>", unsafe_allow_html=True)

        st.markdown("""
                    #### Mr. Rahul Kumar Singh
                    #### Mr. Karnati Sharath Chandra
                    #### Mr. Vishal Vijay Kakade
                    #### Mr. R. Hari Haran
                    #### Mr. Mummareddy Manikumara Swamy
                    #### Mr. Sushilkumar Yadav''')
                    """)
        st.markdown("[ Visit To Github Repositories](https://github.com/RahulSingh409/Project-Document-Classification.git)")
    
    #Horizontal Project selected
    if selected == "Projects":
            st.title(f"You have selected {selected}")
            with st.sidebar:
                image= Image.open("pngwing.com.png")
                add_image=st.image(image,use_column_width=True) 
            
            image2= Image.open("nlp project-2.jpg")
            st.image(image2,use_column_width=True)
            
            st.sidebar.title("Navigation")
            menu_list1 = ['Exploratoriy Data Analysis',"Prediction With Machine Learning"]
            menu_Pre_Exp = st.sidebar.radio("Menu For Prediction & Exploratoriy", menu_list1)
            
            #EDA On Document File
            if menu_Pre_Exp == 'Exploratoriy Data Analysis' and selected == "Projects":
                    st.title('Exploratoriy Data Analysis')

                    df = pd.read_csv('Resume.csv')
                    menu_list2 = ['None', 'Label Analysis','Character & Word Distribution','N-Gram Analysis','Sentiment Analysis','Part Of Speech & Top Common Words','Wordcloud','Named Entity Recognition (NER)','Topic Model Analysis']
                    menu_Exp = st.sidebar.radio("Menu EDA", menu_list2)

                    #Label
                    if menu_Exp == 'None':
                        st.markdown("""
                                    #### Kindly select from left Menu.
                                    """)
                    
                    elif menu_Exp == 'Label Analysis':
                        label_analysis(df)

                    #Distribution    
                    elif menu_Exp == 'Character & Word Distribution':
                        char_word_dist(df)

                    #N-gram Analysis
                    elif menu_Exp == 'N-Gram Analysis':
                        ngram_ana(df)

                    #Sentiment
                    elif menu_Exp == 'Sentiment Analysis':
                        sentiment(df)

                    #POS & Top Words
                    elif menu_Exp == 'Part Of Speech & Top Common Words':
                        pos_common_words(df)

                    #Wordcloud
                    elif menu_Exp == 'Wordcloud':
                        wordcloud_resume(df)

                    #NER
                    elif menu_Exp == 'Named Entity Recognition (NER)':
                        NER2(df)

                    #Topic Modeling
                    elif menu_Exp == 'Topic Model Analysis':
                        preprocess_news(df)

            elif menu_Pre_Exp == "Prediction With Machine Learning" and selected == "Projects":
                    st.title('Prediction With Machine Learning')
                    
                    menu_list3 = ['Checking ML Method And Accuracy' , 'Prediction' ]
                    menu_Pre = st.radio("Menu Prediction", menu_list3)
                    
                    #Checking ML Method And Accuracy
                    if menu_Pre == 'Checking ML Method And Accuracy':
                            st.title('Checking Accuracy On Different Algorithms And Vectorization')
                            df = pd.read_csv('Resume.csv')
                            if st.checkbox("View data"):
                                st.write(df)
                            model = st.selectbox("ML Method",['Logistic Regression', 'XGB Classifier', 'SVM Classifier' , 'Decision Tree Classifier' , 'Random Forest Classifier', 'KNN Classifier' , 'Naive Bayes Classifier'])
                            vector= st.selectbox("Vector Method",[ 'CountVectorizer' , 'TF-IDF'])

                            if st.button('Analyze'):
                                #Logistic Regression & CountVectorizer
                                if model=='Logistic Regression' and vector=='CountVectorizer':
                                    logistic_reg_count(df)
                                    
                                #Logistic Regression & TF-IDF
                                elif model=='Logistic Regression' and vector=='TF-IDF':
                                    logistic_reg_tfiidf(df)

                                #XGB Classifier & CountVectorizer
                                elif model=='XGB Classifier' and vector=='CountVectorizer':
                                    xgb_count(df)                                    
                                
                                #XGB Classifier & TF-IDF
                                elif model=='XGB Classifier' and vector=='TF-IDF':
                                    xgb_tfidf(df)
                                
                                #SVM & CountVectorizer
                                elif model=='SVM Classifier' and vector=='CountVectorizer':
                                    svm_count(df)
                                
                                #SVM & TF-IDF
                                elif model=='SVM Classifier' and vector=='TF-IDF':
                                    svm_tfidf(df)
                                
                                #Decision Tree Classifier & CountVectorizer
                                elif model=='Decision Tree Classifier' and vector=='CountVectorizer':
                                    dec_tree_count(df)
                                
                                #Decision Tree Classifier & TF-IDF
                                elif model=='Decision Tree Classifier' and vector=='TF-IDF':
                                    dec_tree_tfidf(df)
                                
                                #Random Forest Classifier & CountVectorizer
                                elif model=='Random Forest Classifier' and vector=='CountVectorizer':
                                    random_forest_count(df)
                                    
                                #Random Forest Classifier & TF-IDF
                                elif model=='Random Forest Classifier' and vector=='TF-IDF':
                                    random_forest_tfidf(df)
                                
                                #KNN Classifier & CountVectorizer
                                elif model=='KNN Classifier' and vector=='CountVectorizer':
                                    knn_count(df)
                                
                                #KNN Classifier & TF-IDF
                                elif model=='KNN Classifier' and vector=='TF-IDF':
                                    knn_tfidf(df)
                                
                                #Naive Bayes Classifier & CountVectorizer
                                elif model=='Naive Bayes Classifier' and vector=='CountVectorizer':
                                    naive_count(df)
                                
                                #Naive Bayes Classifier & TF-IDF 
                                elif model=='Naive Bayes Classifier' and vector=='TF-IDF':
                                    naive_tfidf(df)
                                
                    elif menu_Pre == 'Prediction':
                        st.title('Prediction')
                        st.markdown("""
                               #### Kindly select from left Menu.
                                    """)
                            
                        menu_list4 = ['Paste/Write Text' , 'Upload File(.docx,.pdf)']
                        menu_Tex_File = st.sidebar.radio("Menu Prediction By Text Or File Upload", menu_list4)
                            
                        
                        if menu_Tex_File == 'Paste/Write Text':
                            message = st.text_area("Enter Text Only PeopleSoft, SQL Developer, React JS, Workday & Internship Resume Accept", "Type Here.....")
                            input2= str(message)
                            input3=[input2]
                            if st.button("Analyze"):
                                text_resume(input3)                                 
                        
                        elif menu_Tex_File == 'Upload File(.docx,.pdf)':
                            st.subheader("DocumentFiles")
                            docx_file = st.file_uploader("Upload File  Only PeopleSoft, SQL Developer, React JS, Workday & Internship Resume Accept",type=['docx','pdf'])
                            if st.button("Process"):
                                if docx_file is not None:
                                    file_details = {"Filename":docx_file.name,"FileType":docx_file.type,"FileSize":docx_file.size}
                                    st.write(file_details)
                                    # Check File Type
                                    if docx_file.type == "text/plain":
                                        # raw_text = docx_file.read() # read as bytes
                                        # st.write(raw_text)
                                        # st.text(raw_text) # fails
                                        st.text(str(docx_file.read(),"utf-8")) # empty
                                        raw_text = str(docx_file.read(),"utf-8") # works with st.text and st.write,used for futher processing
                                        # st.text(raw_text) # Works
                                        st.write(raw_text) # works
                                    elif docx_file.type == "application/pdf":
                                        # raw_text = read_pdf(docx_file)
                                        # st.write(raw_text)
                                        try:
                                            with pdfplumber.open(docx_file) as pdf:
                                                page = pdf.pages[0]
                                                raw_text=page.extract_text()
                                        except:
                                            st.write("None")
                                    
                                    elif docx_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                                        # Use the right file processor ( Docx,Docx2Text,etc)
                                        raw_text = docx2txt.process(docx_file) # Parse in the uploadFile Class directory
                                                                                                
                                convert_text = contractions.fix(raw_text)
                                convert_text= nltk.word_tokenize(convert_text)
                                convert_text= normalize(convert_text)
                                convert_text= lemmatize_verbs(convert_text)
                                            
                                input1 = [convert_text]
                                #loaded_vec.fit(input1)
                                result_pred = clff.predict(loaded_vec.transform(input1))
                                st.success('The predicted LabelId is {}'.format(result_pred))
                                st.markdown("""
                                        #### Data Classification on Resume, LabelId Resemble Below Resume:
                                        + 1. Peoplesoft resumes-           [0]
                                        + 2. SQL Developer resumes-        [1]
                                        + 3. Workday Resumes-              [2]
                                        + 4. React JS-                     [3]
                                        + 5. Internship-                   [4]               
                                        """)
                                if result_pred == 0:
                                    image= Image.open("PeopleSoft_logo.svg.png")
                                    st.image(image,use_column_width=True)
                                elif result_pred == 1:
                                    image= Image.open("sql Developer.png")
                                    st.image(image,use_column_width=True)    
                                elif result_pred == 2:
                                    image= Image.open("workday.png")
                                    st.image(image,use_column_width=True)    
                                elif result_pred == 3:
                                    image= Image.open("ReactJS.png")
                                    st.image(image,use_column_width=True)    
                                elif result_pred == 4:
                                    image= Image.open("internship.png")
                                    st.image(image,use_column_width=True)  

                                                      
if __name__=='__main__':
    main()            
            
            

