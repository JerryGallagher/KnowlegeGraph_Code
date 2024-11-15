#test.py
import nltk
import re

#function
def displayText():
    print("Hello world in test")

def preprocess_code(code):
    cleaned_code = re.sub(r'//.*|/\*.*?\*/', '', code)

    tokens = nltk.word_tokenize(cleaned_code)

    return tokens

source_code = "def my_function(x):\n  return x + 1"
tokens = preprocess_code(source_code)
print(tokens)
