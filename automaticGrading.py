# -*- coding: utf-8 -*-

# Script written by Johanna de Vos, U908153
# Text and multimedia mining
# Automatic grading of open exam questions

# Import modules
import pandas as pd
import gensim
import random
import os
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from gensim import corpora, models
from scipy.stats.stats import pearsonr
from sklearn.model_selection import train_test_split

# Set seed for reproducability of results
random.seed(2017)

# Set working directory
#os.chdir("C:/Users/U908153/Desktop/GitHub/AutomaticGrading")
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
    
    # Split text by newline
    text = text.split("\n")
    
    return text

# Rearrange the data in a dataframe
def create_df(text):
    print("Creating data frame...")
    
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
            grades.append(int(grade))
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
    print("Adding reference answer...")
    ref = pd.Series(["Ref","Ref","Ref",ref_answer_raw,"","",""], index = cols)
    df_ref = df.append(ref, ignore_index = True)
    return df_ref   

# Tokenize, lemmatize, and remove stop words
def tok_lem(df):
    print("Tokenizing, lemmatizing, and removing stop words...")
    
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

# Create dictionary of vocabulary
def dictionary(df):
    print("Creating a dictionary...")
    
    dictionary = corpora.Dictionary(df['Final'])
    
    return dictionary

# Create training, validation and test set
def split(df):
    print("Creating a training, validation and test set...")
    
    ref = df[-1:]
    train, test = train_test_split(df[:-1], test_size = 0.2, random_state = 2017) # Split 80/20, pseudo-random number for reproducability
    return ref, train, test
    
# Implement k-fold cross validation
# https://towardsdatascience.com/train-test-split-and-cross-validation-in-python-80b61beca4b6

# Construct document-term matrices
def doc_term(train, test, ref):
    print("Creating document-term matrices...")
    #print(dictionary.token2id)
    
    # Convert dictionary into bag of words
    dtm_train = [dictionary.doc2bow(text) for text in train['Final']] # Student answers, train split
    dtm_test = [dictionary.doc2bow(text) for text in test['Final']] # Student answers, test split
    dtm_ref = [dictionary.doc2bow(text) for text in ref['Final']] # Reference answer
    dtm_ref = dtm_ref[0]
    
    return dtm_train, dtm_test, dtm_ref

# Generate LDA model
def lda(dictionary, dtm_train):
    print("Training the LDA model...")
    
    ldamod = models.ldamodel.LdaModel(dtm_train, num_topics=2, id2word = dictionary, passes = 20)
    print(ldamod.print_topics(num_topics=2, num_words=5))
    
    return ldamod

# Calculate document similarities with LDA model
def sim(df, dtm_ref, dtm_stu):
    print("Calculating similarity scores...")
    
    sim_scores = []
    counter = 0
    
    for index, row in df.iterrows():
        if dtm_stu[counter]: # If the answer contains text
            sim = gensim.matutils.cossim(ldamod[dtm_ref], ldamod[dtm_stu[counter]])
            print("The cosine similarity is:", sim, "and the grade was:", df['Grade'][index])
            sim_scores.append(sim)
        elif not dtm_stu[counter]: # If the answer is empty
            sim_scores.append(0)
            
        counter += 1
    
    grades = df['Grade'].tolist()
    print(pearsonr(sim_scores, grades))
    
    return sim_scores

# Transform similarity scores into grades
def sim_to_grade(sim_scores):
    print("Transforming similarity scores into grades...")
    
    # Multiply the similarity scores by 10
    pred_grades = [round(sim*10) for sim in sim_scores] 

    return pred_grades

# Evaluate: correlate predicted grades with lecturer-assigned grades
def corr(pred_grades):
    lec_grades = train['Grade'].tolist()
    corr, sig = pearsonr(pred_grades, lec_grades)
    
    return corr
           
# Run code
if __name__ == "__main__":
    
    ref_answer_raw = open_file('referenceAnswer.txt') # Read reference answer
    ref_answer = preprocess(ref_answer_raw) # Preprocess reference answer
    stud_answers_raw = open_file('studentAnswers.txt') # Read student answers
    stud_answers = preprocess(stud_answers_raw) # Preprocess student answers
    df, cols = create_df(stud_answers) # Create dataframe of student answers
    df = add_ref(ref_answer, cols) # Add reference answer to dataframe
    df = tok_lem(df) # Tokenize and lemmatize all answers, and remove stop words
    dictionary = dictionary(df) # Create dictionary of vocabulary
    ref, train, test = split(df) # Split the data into a 80/20 (train/test)
    dtm_train, dtm_test, dtm_ref = doc_term(train, test, ref) # Create dictionary and get frequency counts
    ldamod = lda(dictionary, dtm_train) # Train LDA model on train split
    sim_scores = sim(train, dtm_ref, dtm_train) # Similarity scores for training data
    pred_grades = sim_to_grade(sim_scores)
    corr = corr(pred_grades)







