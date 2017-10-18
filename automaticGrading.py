# -*- coding: utf-8 -*-

# Script written by Johanna de Vos, U908153
# Text and multimedia mining
# Automatic grading of open exam questions

# Import modules
import os
import pandas

# Set working directory
os.chdir('C:/Users/U908153/Desktop/Github/AutomaticGrading')

# Open file
def open_file():
    with open ('AIP_EN.txt') as file:
        raw_text = file.read()
        return raw_text

# Remove irregular white lines
def remove_white(raw_text):
    text = raw_text.replace("\n\nTentamennummer", "\nTentamennummer")
    return text

# Create dataframe
#def create_df(text):
    

# Run code
if __name__ == "__main__":
    raw_text = open_file()
    text = remove_white(raw_text)