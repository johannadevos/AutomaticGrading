# -*- coding: utf-8 -*-

# Script written by Johanna de Vos, U908153
# Text and multimedia mining
# Automatic grading of open exam questions

# Import modules
import os
import pandas as pd
import gensim
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from gensim import corpora, models
from scipy.stats.stats import pearsonr

# Set working directory
#os.chdir('C:/Users/U908153/Desktop/Github/AutomaticGrading')

# Open file
def open_file(file):
    with open (file) as file:
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

    return df, cols

# Add reference answer to dataframe
def add_ref(ref_answer, cols):
    ref = pd.Series(["Ref","Ref","Ref",ref_answer_raw,"","",""], index = cols)
    df_ref = df.append(ref, ignore_index = True)
    return df_ref   

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
    print(dictionary.token2id)
    
    # Convert dictionary into bag of words
    dtm_stu = [dictionary.doc2bow(text) for text in df['Final'][:-1]] # Student answers
    dtm_ref = [dictionary.doc2bow(text) for text in df['Final'][-1:]] # Reference answer
    dtm_ref = dtm_ref[0]
    
    return dictionary, dtm_stu, dtm_ref

# Correlate raw frequencies
def cor_ref_stu(dtm_ref, dtm_stu):

    ID_ref = [i[0] for i in dtm_ref] # List of word IDs in the reference answer
    dtm_stu2 = dtm_stu[:] # Make a copy of the document-term matrix of the students' answers
    correlations = []
    
    for stu in range(len(dtm_stu2)): # For all student answers
        print(stu)
        dtm_ref2 = dtm_ref[:] # Make a copy of the document-term matrix of the reference answer
        
        ID_stu = [i[0] for i in dtm_stu2[stu]] # List of word IDs in the student's answer
    
        for i,j in dtm_ref: # For all IDs and counts in the reference answer
            if not i in ID_stu: # If the ID is not in the student's answer
                dtm_stu2[stu].append((i, 0)) # Append this ID, and give count 0
                
        for i,j in dtm_stu2[stu]: # For all IDs and counts in the student's answer
            if not i in ID_ref: # If the ID is not in the reference answer
                dtm_ref2.append((i, 0)) # Append this ID, and give count 0
        
        dtm_ref2.sort(key=lambda x: x[0]) # Sort the DTM of the reference answer by ID
        dtm_stu2[stu].sort(key=lambda x: x[0]) # Sort the DTM of the student's answer by ID
        
        counts_ref = [i[1] for i in dtm_ref2] # Extract the counts from the reference answer
        counts_stu = [i[1] for i in dtm_stu2[stu]] # Extract the counts from the student's answer
        
        correlation = pearsonr(counts_ref, counts_stu) # Calculate correlations
        correlations.append(correlation[0]) # Correlation[1] is the p-value
        
        co_sim = gensim.matutils.cossim(counts_ref, counts_stu)                   
        print(co_sim)
        
    return correlations

# Generate LDA model
def lda(dictionary, corpus):
    ldamodel = models.ldamodel.LdaModel(corpus, num_topics=5, id2word = dictionary, passes = 5)
    print(ldamodel.print_topics(num_topics=5, num_words=5))
    return ldamodel

# Calculate document similarities with LDA model
sim = gensim.matutils.cossim(vec_lda1, vec_lda2)

# Run code
if __name__ == "__main__":
    
    ref_answer_raw = open_file('referenceAnswer.txt') # Read reference answer
    ref_answer = preprocess(ref_answer_raw) # Preprocess reference answer
    stud_answers_raw = open_file('studentAnswers.txt') # Read student answers
    stud_answers = preprocess(stud_answers_raw) # Preprocess student answers
    df, cols = create_df(stud_answers) # Create dataframe of student answers
    df = add_ref(ref_answer, cols) # Add reference answer to dataframe
    df = tok_lem(df) # Tokenize and lemmatize all answers, and remove stop words
    dictionary, dtm_stu, dtm_ref = doc_term(df) # Create dictionary and get frequency counts
    #correlations = cor_ref_stu(dtm_ref, dtm_stu)
    ldamodel = lda(dictionary, dtm_stu) # Train LDA model on student data

   