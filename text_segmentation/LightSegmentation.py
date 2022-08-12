import re


# Basic tokenization
def split_into_tokens(text):
    # TODO: handle era strings
    pattern_words = re.compile("\w*")
    words = text.split()
    tokens = []
    for word in words:
        for match in re.finditer(pattern_words, word):
            if match is None:
                break
            group = match.group()
            if group != "":
                tokens.append(group)
    return tokens


def split_into_sentences(article):
    sentences = []
    punctuation = [".", "!", "?"]
    sentence = ""
    for i, c in enumerate(article):
        sentence += c
        if c in punctuation and 0 < i < (len(article) - 2):
            # anglos use dots as decimal separators
            if not (article[i - 1].isdigit() and article[i + 1].isdigit()):
                # dots denoting the end of a sentence are mostly followed by certain characters
                if article[i + 1] in [" ", "\n", "\""]:
                    # next sentence should start with capitalized letter
                    if not article[i + 2].isalpha() or article[i + 2].isupper():
                        sentences.append(sentence.strip())
                        sentence = ""
    if sentence.strip() != "":
        sentences.append(sentence.strip())
    return sentences
