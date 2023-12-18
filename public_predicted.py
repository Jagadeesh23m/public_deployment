# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 12:52:10 2023

@author: shiva
"""

import numpy as np
import pickle
import streamlit as st

model=pickle.load(open('svm_model.sav','rb'))

vector=pickle.load(open('vectarizer.sav','rb'))


def predict_review(review):
    #comm=['this hotel is just ok']
    vec=vector.transform(review).toarray()
    pred=model.predict(vec)
    #print(pred)
    if (pred[0]==2):
        return 'the review is positive :slightly_smiling_face:'
    elif(pred[0]==0):
        return 'the review is negative :slightly_frowning_face:'
    else:
        return 'the review is neutral :neutral_face:'

def main():
    
    st.title('Sentiment Analysis')
    
    rev=st.text_input('Enter the review')
    
    sentiment=''
    
    if st.button('predict'):
        sentiment=predict_review([rev])
    st.success(sentiment)    

if __name__=='__main__':
    main()