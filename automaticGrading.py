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
from gensim import corpora, models, utils
from scipy.stats.stats import pearsonr
from scipy.sparse import csr_matrix
from scipy import spatial
import sklearn.model_selection
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from matplotlib.pyplot import plot, draw, show

import matplotlib.pyplot as plt
from collections import Counter
from ggplot import *

# Set seed for reproducability of results
random.seed(2017)

# Set working directory
os.chdir("C:/Users/johan/Documents/GitHub/AutomaticGrading")
#os.chdir("C:/Users/U908153/Desktop/GitHub/AutomaticGrading")

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
    print("Creating data frame for the student answers...")
    
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
    print("Creating data frame for the text book...")
    
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

# TO DO: remove numbers
# TO DO: standardise British/American spelling?

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
    print("Training the LDA model...\n")
    
    ldamod = models.ldamodel.LdaModel(dtm_train, num_topics=6, id2word = dictio, passes = 20)
    #print("This is the LDA model:\n")
    #print(ldamod.print_topics(num_topics=6, num_words=2))
    
    return ldamod

# Generate LSA model
def lsa(dictio, dtm_train):
    print("Training the LSA model...\n")
    
    lsamod = models.LsiModel(dtm_train, id2word=dictio, num_topics=20, chunksize=1, distributed=False) 
    #print("This is the LSA model:\n")
    #print(lsamod.print_topics(num_topics=20, num_words=5))
    
    return lsamod

# Calculate document similarities based on TF-IDF
def sim_baseline(tfidf_train, tfidf_ref):
    print("Calculating similarity scores for the baseline TF-IDF model...")
    
    sim_scores = []
    counter = 0
    
    for row in range(tfidf_train.shape[0]):
        if tfidf_train[row].getnnz() > 0: # If the answer contains text
            sim = 1 - spatial.distance.cosine(tfidf_train[row].toarray(), tfidf_ref.toarray()) 
            sim_scores.append(sim) 
        else:
            sim_scores.append(0) 
        counter += 1
    
    return sim_scores

# Calculate document similarities with a topic model
def sim_topic_mod(model, dtm_train, dtm_ref):
    print("\nCalculating similarity scores...\n")
    
    sim_scores = []
    counter = 0
    
    for answer in dtm_train:
        if len(answer) > 0: # If the answer contains text
            sim = gensim.matutils.cossim(model[dtm_ref], model[answer])
            sim_scores.append(sim) 
        else:
            sim_scores.append(0)
        counter += 1
    
    return sim_scores

# Transform similarity scores into grades
def sim_to_grade(sim_scores):
    print("Transforming similarity scores into grades...\n")
    
    # Multiply the similarity scores by 10
    pred_grades = [round(sim*10) for sim in sim_scores] 

    return pred_grades

def tfidf(train, ref):
    
    # Temporarily merge 'train' and 'ref' into one dataframe
    df_train_ref = train.append(ref)
    
    # Get document-term matrix (TF-IDF)    
    tfidf_mod = TfidfVectorizer(analyzer='word', min_df = 0) # Set up model
    strings = [" ".join(word) for word in df_train_ref['NoStops']] # Transform answers from a list of words to a string
    tfidf = tfidf_mod.fit_transform(strings) # Get TF-IDF matrix                                   
    tfidf_train = tfidf[:-1]
    tfidf_ref = tfidf[-1]
    
    '''
    print(tfidf_train)
    tfidf_train.A # Shows array, same as tfidf_train.toarray()
    tfidf_train.indices # 
    tfidf_train.data
    tfidf_train.indptr
    '''
    
    return tfidf_train, tfidf_ref

def evaluate(pred_grades, val_grades):
    # Get correlation between predicted grades (validation set) and lecturer-assigned grades 
    # TO DO: exclude empty answers from this calculation, as they improve the score without improving the algorithm
    # (Empty answers are always scores as 0)
    correl, sig = pearsonr(pred_grades, val_grades)
    print("The correlation between the predicted and real grades is:", correl, "\n")
    
    # Plot the predicted grades and lecturer-assigned grades
    df_grades = pd.DataFrame({'Predicted': pred_grades, 'Assigned': val_grades})
    df_counts = df_grades.groupby(['Predicted', 'Assigned']).size().reset_index(name='Count')
    df_counts.plot(kind='scatter', x='Predicted', y='Assigned', s=df_counts['Count'], xlim=[-0.3,10.3])
    show()
     
    return correl

# Training a topic model on the student data, using k-fold cross-validation
def topic_mod_cross_val(train, ref, tfidf_ref, dictio, topic_mod="LDA", counting="TF-IDF"):
    
    # Get document-term matrix for the reference answer (raw counts)
    raw_counts_ref = [dictio.doc2bow(text) for text in ref['NoStops']][0]
      
    # Stratified 10-fold cross-validation (stratificiation is based on the real grades)
    splits = 2
    skf = sklearn.model_selection.StratifiedKFold(n_splits=splits) # TO DO: What train-test split is used? (e.g. 80/20)
    
    # Get indices of columns in dataframe
    index_NoStops = train.columns.get_loc("NoStops")
    index_Grade = train.columns.get_loc("Grade")
    
    # Create empty list to store the correlations of the 10 folds
    all_corrs = []
     
    # Start 10-fold cross validation
    for fold_id, (train_indexes, validation_indexes) in enumerate(skf.split(train, train.Grade)):
        
        # Print the fold number
        print("\nFold %d" % (fold_id + 1), "\n")
         
        # Collect the data for this train/validation split
        train_texts = [df.iloc[x, index_NoStops] for x in train_indexes]
        train_grades = [df.iloc[x, index_Grade] for x in train_indexes]
        val_texts = [df.iloc[x, index_NoStops] for x in validation_indexes]
        val_grades = [df.iloc[x, index_Grade] for x in validation_indexes]
        
        # Get the document-term matrices (raw counts)
        raw_counts_train = [dictio.doc2bow(text) for text in train_texts] # Student answers, train split        
        raw_counts_val = [dictio.doc2bow(text) for text in val_texts] # Student answers, validation split
        
        if topic_mod == "LDA":
            model = lda(dictio, raw_counts_train) 
            
        elif topic_mod == "LSA":
            
            if counting == "raw":
                model = lsa(dictio, raw_counts_train)
                
            elif counting == "TF-IDF":
            
                # Get the document-term matrices (TF-IDF)
                tfidf_mod = TfidfVectorizer(analyzer='word', min_df = 0) # Set up model
                strings = [" ".join(word) for word in train_texts] # Transform answers from a list of words to a string
                tfidf_train = tfidf_mod.fit_transform(strings) # Get TF-IDF matrix
                
                # Tranform TF-IDF matrix into list of tuples: (dictio index, TF-IDF score)
                tfidf_tuples = []
                
                for row in tfidf_train:
                    z = zip(row.indices, row.data)
                    tfidf_tuples.append(list(z))
                        
                # Train LSA model 
                model = lsa(dictio, tfidf_tuples) 

        # Get similarity scores of validation answers to reference answer
        sim_scores = sim_topic_mod(model, raw_counts_val, raw_counts_ref)
        
        # Transform similarity scores into grades
        # TO DO: use the training set to try out different mapping algorithms?
        pred_grades = sim_to_grade(sim_scores)
        
        # Evaluate
        corr = evaluate(pred_grades, val_grades)
        all_corrs.append(corr)
          
    # Average correlation over 10 folds
    av_corr = sum(all_corrs) / len(all_corrs)
    print("The average correlation over", splits, "folds is:", av_corr)

# Training an LDA on a psychology text book, and using this model to predict grades on the training set
def lda_book(df_book, train, ref):
    
        # Get a list of all sentences in the book
        book_texts = list(df_book['NoStops'])
        
        # Get the document-term matrix for the book       
        dictio_book = dictionary(df_book) # Create dictionary of vocabulary
        raw_counts_book = [dictio_book.doc2bow(text) for text in book_texts] 
        
        # Train LDA model
        ldamod_book = lda(dictio_book, raw_counts_book)

        # Document-term matrix for student answers
        raw_counts_train = [dictio_book.doc2bow(text) for text in train['NoStops']] # Student answers, train split

        # Document-term matrix for reference answer
        raw_counts_ref = [dictio_book.doc2bow(text) for text in ref['NoStops']][0] # Reference answer
        
        # Get similarity scores of training answers to reference answer
        # TO DO: similarity scores are unreasonably high (often around .99)
        # As a result, only grades of 10 are predicted
        sim_scores_lda = sim_topic_mod(ldamod_book, raw_counts_train, raw_counts_ref)
        
        # Transform similarity scores into grades
        # TO DO: use the training set to try out different mapping algorithms?
        pred_grades = sim_to_grade(sim_scores_lda)
        
        # Get assigned grades
        val_grades = list(train["Grade"])
        
        # Get correlation between predicted grades (validation set) and lecturer-assigned grades 
        # TO DO: exclude empty answers from this calculation, as they improve the score without improving the algorithm
        # (Empty answers are always scored as 0)
        evaluate(pred_grades, val_grades)

    
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
    tfidf_train, tfidf_ref = tfidf(train, ref) # Get TF-IDF document-term matrix
    
    # Read and prepare Psychology book
    book_raw = open_file('psyBook.txt') # Open book
    book_raw = book_raw[:50000] # To try new things out without having to wait very long
    book = book_raw.replace("\n", " ") # Remove white lines
    book = book.replace(chr(0x00AD), "") # Remove soft hyphens
    book = preprocess(book) # Preprocess book
    sent_book = sent_tokenize(book) # Split into sentences
    df_book, cols_book = create_df_book(sent_book) # Create dataframe 
    df_book = tok_lem(df_book) # Tokenize, lemmatize, remove stop words

    # Baseline model
    sim_scores_baseline = sim_baseline(tfidf_train, tfidf_ref)
    pred_grades = sim_to_grade(sim_scores_baseline)
    val_grades = list(train["Grade"])
    evaluate(pred_grades, val_grades)
        
    # LDA model: train on student answers
    topic_mod_cross_val(train, ref, None, dictio, topic_mod="LDA")
  
    # LDA model: train on Psychology book
    lda_book(df_book, train, ref) 
    
    # LSA model: train on student answers
    topic_mod_cross_val(train, ref, tfidf_ref, dictio, topic_mod="LSA", counting="TF-IDF")
    
    # LSA model: train on Psychology book














