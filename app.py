import pandas as pd
import numpy as np
import re 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from pickle import load
import streamlit as st
from sklearn.feature_extraction.text import TfidfTransformer

nlp=spacy.load("en_core_web_sm")
wordnet=WordNetLemmatizer()
stop_words=stopwords.words('english')

not_stopwords = ("aren", "aren't", "couldn", "couldn't", "didn", "didn't",
                 "doesn", "doesn't", "don", "don't", "hadn", "hadn't", "hasn",
                 "hasn't", "haven", "haven't", "isn", "isn't", "mustn",
                 "mustn't", "no", "not", "only", "shouldn", "shouldn't",
                 "should've", "wasn", "wasn't", "weren", "weren't", "will",
                 "wouldn", "wouldn't", "won't", "very")
stop_words_ = [words for words in stop_words if words not in not_stopwords]
stop_words_.append("I")
stop_words_.append("the")
stop_words_.append("s")
stop_words_.extend([
    "will", "always", "go", "one", "very", "good", "only", "mr", "lot", "two",
    "th", "etc", "don", "due", "didn", "since", "nt", "ms", "ok", "almost",
    "put", "pm", "hyatt", "grand", "till", "add", "let", "hotel", "able",
    "per", "st", "couldn", "yet", "par", "hi", "well", "would", "I", "the",
    "s", "also", "great", "get", "like", "take", "thank"
])
    
def Prediction(corpus):
    output1 = []
    review =str(corpus)
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    review = review.split()
    review = nlp(' '.join(review))
    review = [token.lemma_ for token in review]
    review = [word for word in review if word not in stop_words_]
    review = ' '.join(review)
    output1.append(review)
    
    loaded_TFIDF = load(open('model_TFIDF.sav', 'rb'))
    X = pd.DataFrame((loaded_TFIDF.transform(output1)).toarray())
    
    loaded_model = load(open('finalized_model.sav','rb'))
    pred = int(loaded_model.predict(X))
    if pred == 0:
        return 'Negative'
    elif pred == 1: 
        return 'Positive'

def Keywords(corpus):
    output2 = []
    review =str(corpus)
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    review = review.split()
    review = nlp(' '.join(review))
    review = [token.lemma_ for token in review]
    review = [word for word in review if word not in stop_words_]
    review = ' '.join(review)
    output2.append(review)
    
    tfidf2 = TfidfVectorizer(norm="l2",analyzer='word', stop_words=stop_words_,ngram_range=(1,2))
    tfidf2_x = tfidf2.fit_transform(output2)
    tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf_transformer.fit(tfidf2_x)
    
    feature_names = tfidf2.get_feature_names()
    tf_idf_vector = tfidf_transformer.transform(tfidf2.transform(output2))
    
    def sort_coo(coo_matrix):
        tuples = zip(coo_matrix.col, coo_matrix.data)
        return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

    sorted_items=sort_coo(tf_idf_vector.tocoo())
    def extract_topn_from_vector(feature_names, sorted_items, topn=10):
            """get the feature names and tf-idf score of top n items"""
    
            sorted_items = sorted_items[:topn]
 
            score_vals = []
            feature_vals = []

            for i, score in sorted_items:
                score_vals.append(round(score, 3))
                feature_vals.append(feature_names[i])
            results= feature_vals
    
            return pd.Series (results)
 
    attributes=extract_topn_from_vector(feature_names,sorted_items,10)
    return attributes  

def main():
    html_temp = """ 
    <div style = "background-color:white; padding-bottom:10px"> 
    <h1 style = "color:black;text-align:center;">INF1002 Project 6:<br>Hotel Review Sentiment Analysis</h1>
    </div> 
    """
    # image banner
    col1, col2, col3 = st.columns(3)
    with col1, col3:
        st.write('')
    with col2:
        st.image("sit_logo.png", width=240)
    # load frontend
    st.markdown(html_temp, unsafe_allow_html=True)
    st.subheader("Based on reviewer's comments, evaluate hotel service.")
    text = st.text_input('Enter text here')  
    if st.button("Predict"):
        predict = Prediction(text)
        if predict == 'Positive':
            st.success('The sentiment of the review is {}'.format(predict))
        elif predict == 'Negative':
            st.error('The sentiment of the review is {}'.format(predict))
        st.subheader("Review Attributes")
        keytxt = Keywords(text)
        for i in keytxt:
            if predict == 'Positive':
                st.success(i)
            elif predict == 'Negative':
                st.error(i)
    
if __name__ == '__main__':
    main()