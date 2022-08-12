import re
import torch
import numpy as np
import LightSegmentation
from numpy.linalg import norm
from scipy import stats
from utils.Roberta import Roberta

roberta = Roberta.instance()


# call this from outside
def get_segmentation(text):
    sentences = LightSegmentation.split_into_sentences(text)
    segments = generate_segments(sentences)
    return segments


# retrieves embedding of the sequence from the model
def get_words_embedding(text):
    lhs = roberta.model(**roberta.tokenizer(text, return_tensors="pt")).last_hidden_state
    return lhs.reshape(-1, lhs.size(2))


# calculates fixed-size sentence embedding by taking the average of every feature vector element across all words
def get_av_sentence_embedding(words_embedding):
    sentence_embedding = torch.zeros(words_embedding.size(1), dtype=words_embedding.dtype)
    for i in range(words_embedding.size(0)):
        sentence_embedding += words_embedding[i]
    sentence_embedding /= words_embedding.size(0)
    return sentence_embedding.detach().numpy()


# calculates sentence embedding by taking the max value of each feature vectors element across all words
def get_max_sentence_embedding(words_embedding):
    sentence_embedding = torch.zeros(words_embedding.size(1), dtype=words_embedding.dtype)
    for feature in range(words_embedding.size(1)):
        feature_max = words_embedding[0][feature]
        for word in range(words_embedding.size(0)):
            if words_embedding[word][feature] > feature_max:
                feature_max = words_embedding[word][feature]
        sentence_embedding[feature] = feature_max
    return sentence_embedding.detach().numpy()


# cosine similarity
def get_cos_sim(vec1, vec2):
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))


# group sentences into segments
def generate_segments(sentences):
    sims = []
    for i in range(len(sentences) - 1):
        vec1 = get_av_sentence_embedding(get_words_embedding(sentences[i]))
        vec2 = get_av_sentence_embedding(get_words_embedding(sentences[i + 1]))
        similarity = get_cos_sim(vec1, vec2)
        sims.append(similarity)
    distribution = stats.norm.fit(sims)
    threshold = distribution[0] - distribution[1]
    segments = []
    segment = sentences[0]
    for i in range(len(sentences) - 1):
        if sims[i] > threshold:
            segment += " " + sentences[i + 1]
        else:
            segments.append(segment.strip())
            segment = sentences[i + 1]
    if segment.strip() != "":
        segments.append(segment)
    return segments
