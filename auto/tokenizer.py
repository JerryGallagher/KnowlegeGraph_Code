import nltk

#First attempt tokenize source code
def tokenizer(source_file):
    tokens = nltk.word_tokenize(source_file)

    return (tokens)