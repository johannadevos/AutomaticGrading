# -*- coding: utf-8 -*-

# Script written by Johanna de Vos, U908153
# Text and multimedia mining
# Automatic grading of open exam questions

# Import modules
import os
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
#from gensim import corpora, models
import gensim
#from nltk import word_tokenize, sent_tokenize 

# Set working directory
#os.chdir('C:/Users/U908153/Desktop/Github/AutomaticGrading')

# Open file
def open_file():
    with open ('AIP_EN.txt') as file:
        raw_text = file.read()
        
        return raw_text

# Preprocess text
def preprocess(raw_text):
    
    # For empty anwers, insert a dash
    text = raw_text.replace("Antwoord:\n", "Antwoord: -\n")
        
    # Remove white space
    text = text.replace("\\n", "")
    text = text.replace(" /n ", "")
    text = text.replace("/n", "")
    text = text.replace("  ", " ")
    text = text.replace("\n\nTentamennummer", "\nTentamennummer")
    
    # Replace curly quotes
    text = text.replace(chr(0x2019), chr(0x0027)) # Replace right single curly quote with straight quote
    text = text.replace(chr(0x2018), chr(0x0027)) # Replace left single curly quote with straight quote
    text = text.replace(chr(0x201D), chr(0x0022)) # Replace right double curly quotes with straight quotes
    text = text.replace(chr(0x201C), chr(0x0022)) # Replace left double curly quotes with straight quotes

    # Replace abbreviated verbs
    text = text.replace("can't", "cannot")
    text = text.replace("n't", " not")
    
    # Split text by newline
    text = text.split("\n")
    
    return text

# Rearrange the data in a dataframe
def create_df(text):
    exam_numbers = []
    subject_codes = []
    grades = []
    answers = []
    
    # Extract information from running text
    for i in range(len(text)):
        if i%4 == 0:
            exam_number = text[i][16:]
            exam_numbers.append(exam_number)
        elif i%4 == 1:
            subject_code = text[i][13:]
            subject_codes.append(subject_code)
        elif i%4 == 2:
            grade = text[i][7:]
            grades.append(grade)
        elif i%4 == 3:
            answer = text[i][10:]
            answers.append(answer)
            
    # Create dataframe
    df = pd.DataFrame({'Exam number': exam_numbers, 'Subject code': subject_codes, 'Grade': grades, 'Answer': answers})       
    
    # Add empty columns that can later contain tokenized and lemmatized data
    df['Tokenized'] = ""
    df['Lemmatized'] = ""
    df['Final'] = "" 
    
    # Change order of columns
    cols = ['Subject code', 'Exam number', 'Grade', 'Answer', 'Tokenized', 'Lemmatized', 'Final']
    df = df[cols]    

    return df

# Tokenize, lemmatize, and remove stop words
def tok_lem(df):
    
    # Set up tokenizer and lemmatizer
    tokenizer = RegexpTokenizer(r'\w+')
    lemmatizer = WordNetLemmatizer()    
    
    for i in range(len(df)):
        answer = df['Answer'][i]
        
        # Preprocess
        answer = answer.replace("'s", "") # Remove possessive 's
        answer = answer.lower()
                               
        # Tokenize
        #tok_answer = word_tokenize(answer)
        tok_answer = tokenizer.tokenize(answer) # also removes apostrophe
        df['Tokenized'][i] = tok_answer
        
        # Lemmatize
        lem_answer = []
        
        for word in tok_answer:
            lemma = lemmatizer.lemmatize(word, pos = 'v') # default POS is 'n'
            
            # Hand-crafted rules for words not handled well by lemmatizer
            if lemma.startswith("whorf"):
                lemma = "whorf"
            
            lem_answer.append(lemma)
        
        df['Lemmatized'][i] = lem_answer

        # Remove stop words
        stopped_lemmas = [i for i in lem_answer if not i in stopwords.words('english')]
        df['Final'][i] = stopped_lemmas
        
    return df

# Construct document-term matrix
def doc_term(df):
    dictionary = corpora.Dictionary(df['Final'])
    
    # Convert dictionary into bag of words
    corpus = [dictionary.doc2bow(text) for text in df['Final']]

    return dictionary, corpus

# Generate LDA model
def lda(dictionary, corpus):
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=2, id2word = dictionary, passes = 5)
    print(ldamodel.print_topics(num_topics=2, num_words=5))
    return ldamodel

# Run code
if __name__ == "__main__":
    raw_text = open_file()
    text = preprocess(raw_text)
    df = create_df(text)
    df = tok_lem(df)
    dictionary, corpus = doc_term(df)
    ldamodel = lda(dictionary, corpus)


