# -*- coding: utf-8 -*-

# Script written by Johanna de Vos, U908153
# In the course Text and Multimedia Mining, Radboud University
# Automatic grading of open exam questions

# Import modules
import pandas as pd
import gensim
import random
import os
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from gensim import corpora, models
from scipy.stats.stats import pearsonr
import sklearn.model_selection
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

import matplotlib.pyplot as plt
from collections import Counter
from ggplot import *

# Set seed for reproducability of results
random.seed(2017)

# Set working directory
os.chdir("C:/Users/johan/Documents/GitHub/AutomaticGrading")

# Open file
def open_file(file):
    print("Opening file...")
    
    with open (file) as file:
        raw_text = file.read()
        
        return raw_text

# Preprocess text
def preprocess(raw_text):
    print("Preprocessing raw text...")
    
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
    
    return text

# Rearrange the students' answers in a dataframe
def create_df(text):
    print("Creating data frame...")
    
    exam_numbers = []
    subject_codes = []
    grades = []
    answers = []
    
    # Split text by newline
    text = text.split("\n")
    
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
            grades.append(int(grade))
        elif i%4 == 3:
            answer = text[i][10:]
            answers.append(answer)
            
    # Create dataframe
    df = pd.DataFrame({'ExamNumber': exam_numbers, 'SubjectCode': subject_codes, 'Grade': grades, 'Answer': answers})      
    
    # Add empty columns that can later contain tokenized and lemmatized data
    df['Tokenized'] = ""
    df['Lemmatized'] = ""
    df['NoStops'] = "" 
    
    # Change order of columns
    cols = ['SubjectCode', 'ExamNumber', 'Grade', 'Answer', 'Tokenized', 'Lemmatized', 'NoStops']
    df = df[cols]    

    return df, cols

# Add reference answer to dataframe
def add_ref(ref_answer, cols):
    print("Adding reference answer...")
    ref = pd.Series(["Ref","Ref","Ref",ref_answer_raw,"","",""], index = cols)
    df_ref = df.append(ref, ignore_index = True)
    return df_ref   

# Create dataframe for other input texts
def create_df_book(text):
    print("Creating data frame...")
    
    # Create dataframe
    df = pd.DataFrame({'Answer': text})
    
    # Add empty columns that can later contain tokenized and lemmatized data
    df['Tokenized'] = ""
    df['Lemmatized'] = ""
    df['NoStops'] = "" 
    
    # Change order of columns
    cols = ['Answer', 'Tokenized', 'Lemmatized', 'NoStops']
    df = df[cols]    
    
    return df, cols

# Tokenize, lemmatize, and remove stop words
def tok_lem(df):
    print("Tokenizing, lemmatizing, and removing stop words...")
    
    # Make SubjectCode the index of the dataframe
    #df = df.set_index("SubjectCode") --> with this line, preprocessing the book no longer works
    
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
        df['NoStops'][i] = stopped_lemmas
        
    return df

# Create dictionary of vocabulary
def dictionary(df):
    print("Creating a dictionary...")
    
    dictionary = corpora.Dictionary(df['NoStops'])
    #print(dictionary.token2id)
    
    return dictionary

# Create training, validation and test set
def split(df):
    print("Creating a training, validation and test set...")
    
    ref = df[-1:]
    train, test = train_test_split(df[:-1], test_size = 0.2, random_state = 2017) # Split 80/20, pseudo-random number for reproducability
    return ref, train, test

# Generate LDA model
def lda(dictio, dtm_train):
    print("Training the LDA model...")
    
    ldamod = models.ldamodel.LdaModel(dtm_train, num_topics=6, id2word = dictio, passes = 20)
    print("This is the LDA model:")
    print(ldamod.print_topics(num_topics=6, num_words=2))
    
    return ldamod

# Calculate document similarities with LDA model
def sim(ldamod, dtm_stu, dtm_ref):
    print("Calculating similarity scores...")
    
    sim_scores = []
    counter = 0
    
    for answer in dtm_stu:
        if len(answer) > 0: # If the answer contains text
            sim = gensim.matutils.cossim(ldamod[dtm_ref], ldamod[answer])
            sim_scores.append(sim) 
        else:
            sim_scores.append(0) 
        counter += 1
    
    return sim_scores

# Transform similarity scores into grades
def sim_to_grade(sim_scores):
    print("Transforming similarity scores into grades...")
    
    # Multiply the similarity scores by 10
    pred_grades = [round(sim*10) for sim in sim_scores] 

    return pred_grades

# Training an LDA on the exam data, and using this model to predict grades on a validation set
# (10-fold cross-validation)
def cross_val_lda(train, ref, dictio):

    # Document-term matrix for reference answer --> raw counts
    dtm_ref = [dictio.doc2bow(text) for text in ref['NoStops']] # Reference answer
    dtm_ref = dtm_ref[0]
    
    # Document-term matrix for reference answer --> TF-IDF
    
    
    
    # Stratified 10-fold cross-validation (stratificiation is based on the real grades)
    skf = sklearn.model_selection.StratifiedKFold(n_splits=2)
    
    # Get indices of columns in dataframe
    index_NoStops = df.columns.get_loc("NoStops")
    index_Grade = df.columns.get_loc("Grade")
    
    # Create empty list to store the correlations of the 10 folds
    all_corrs = list()
     
    # Start 10-fold cross validation
    for fold_id, (train_indexes, validation_indexes) in enumerate(skf.split(train, train.Grade)):
        
        # Print the fold number
        print("Fold %d" % (fold_id + 1), "\n")
        
        # Collect the data for this train/validation split
        train_texts = [df.iloc[x, index_NoStops] for x in train_indexes]
        train_grades = [df.iloc[x, index_Grade] for x in train_indexes]
        val_texts = [df.iloc[x, index_NoStops] for x in validation_indexes]
        val_grades = [df.iloc[x, index_Grade] for x in validation_indexes]
        
        # Get the document-term matrices
        dtm_train = [dictio.doc2bow(text) for text in train_texts] # Student answers, train split        
        dtm_val = [dictio.doc2bow(text) for text in val_texts] # Student answers, validation split
        
        # Train LDA model on train split
        ldamod = lda(dictio, dtm_train) 

        # Get similarity scores of validation answers to reference answer
        # TO DO: similarity scores are unreasonably high (often around .99)
        # As a result, only grades of 10 are predicted
        sim_scores = sim(ldamod, dtm_val, dtm_ref)
        print(sim_scores)
        
        # Transform similarity scores into grades
        # TO DO: use the training set to try out different mapping algorithms?
        pred_grades = sim_to_grade(sim_scores)
        
        # Get correlation between predicted grades (validation set) and lecturer-assigned grades 
        # TO DO: exclude empty answers from this calculation, as they improve the score without improving the algorithm
        # (Empty answers are always scores as 0)
        correl, sig = pearsonr(pred_grades, val_grades)
        all_corrs.append(correl)
        
        print("The correlation between the predicted and real grades is:", correl, "\n")
        
        # Plot the predicted grades and lecturer-assigned grades
        df_grades = pd.DataFrame({'Predicted': pred_grades, 'Assigned': val_grades})
        df_counts = df_grades.groupby(['Predicted', 'Assigned']).size().reset_index(name='Count')
        df_counts.plot(kind='scatter', x='Predicted', y='Assigned', s=df_counts['Count'])
       
    # Average correlation over 10 folds
    av_corr = sum(all_corrs) / len(all_corrs)
    print("The average correlation over 10 folds is:", av_corr)

# Training an LDA on a psychology text book, and using this model to predict grades on the training set
def lda_book(df_book, train, ref):
    
        # Get a list of all sentences in the book
        book_texts = list(df_book['NoStops'])
        
        # Get the document-term matrix for the book       
        dictio_book = dictionary(df_book) # Create dictionary of vocabulary
        dtm_book = [dictio.doc2bow(text) for text in book_texts] 
        
        # Train LDA model
        ldamod_book = lda(dictio_book, dtm_book)

        # Document-term matrix for student answers
        dtm_train = [dictio_book.doc2bow(text) for text in train['NoStops']] # Student answers, train split

        # Document-term matrix for reference answer
        dtm_ref = [dictio_book.doc2bow(text) for text in ref['NoStops']] # Reference answer
        dtm_ref = dtm_ref[0]
    
        # Get similarity scores of training answers to reference answer
        # TO DO: similarity scores are unreasonably high (often around .99)
        # As a result, only grades of 10 are predicted
        sim_scores = sim(ldamod_book, dtm_train, dtm_ref)
        print(sim_scores)
        
        # Transform similarity scores into grades
        # TO DO: use the training set to try out different mapping algorithms?
        pred_grades = sim_to_grade(sim_scores)
        
        # Get assigned grades
        val_grades = list(train["Grade"])
        
        # Get correlation between predicted grades (validation set) and lecturer-assigned grades 
        # TO DO: exclude empty answers from this calculation, as they improve the score without improving the algorithm
        # (Empty answers are always scores as 0)
        correl, sig = pearsonr(pred_grades, val_grades) 
        print("The correlation between the predicted and real grades is:", correl, "\n")
        
        # Plot the predicted grades and lecturer-assigned grades
        df_grades = pd.DataFrame({'Predicted': pred_grades, 'Assigned': val_grades})
        df_counts = df_grades.groupby(['Predicted', 'Assigned']).size().reset_index(name='Count')
        df_counts.plot(kind='scatter', x='Predicted', y='Assigned', s=df_counts['Count'])  


# Run code
if __name__ == "__main__":
    
    # Read and prepare student data
    ref_answer_raw = open_file('referenceAnswer.txt') # Read reference answer
    ref_answer = preprocess(ref_answer_raw) # Preprocess reference answer
    stud_answers_raw = open_file('studentAnswers.txt') # Read student answers
    stud_answers = preprocess(stud_answers_raw) # Preprocess student answers
    df, cols = create_df(stud_answers) # Create dataframe of student answers
    df = add_ref(ref_answer, cols) # Add reference answer to dataframe
    df = tok_lem(df) # Tokenize and lemmatize all answers, and remove stop words
    dictio = dictionary(df) # Create dictionary of vocabulary
    ref, train, test = split(df) # Split the data into a 80/20 (train/test)
    
    # Train an LDA model on the student data and predict grades
    cross_val_lda(train, ref, dictio)
       
    # Read and prepare Psychology book
    book_raw = open_file('psyBook.txt') # Open book
    #book_test = book_raw[:50000]
    book = book_raw.replace("\n", " ") # Remove white lines
    book = book.replace(chr(0x00AD), "") # Remove soft hyphens
    book = preprocess(book) # Preprocess book
    sent_book = sent_tokenize(book) # Split into sentences
    df_book, cols_book = create_df_book(sent_book) # Create dataframe 
    df_book = tok_lem(df_book) # Tokenize, lemmatize, remove stop words
    
    # Train an LDA model on the book and predict grades  
    lda_book(df_book, train, ref) 
    
    #TO DO: implement tf-idf
    # https://gist.github.com/clemsos/7692685
    
    #TO DO: implement LSA
    
    



