import random
import datefinder
from segment_labeling import Date
from date_extraction import StringAutomaton
from text_segmentation import LightSegmentation


# Assign label to every segment
def label(segments):
    labeling = []
    for segment in segments:
        # change xyz_label_segment to change date extraction method
        labeling.append((segment, wsa_label_segment(segment)))
    return labeling


def random_label_segment(segment):
    return Date.Date(random.randint(0, 2022), random.randint(1, 12), random.randint(1, 28))


# Date extraction using datefinder for reference
def df_label_segment(segment):
    matches = datefinder.find_dates(segment)
    match = None
    for m in matches:
        match = m
        break
    if match is not None:
        return Date.Date(match.year, match.month, match.day)
    else:
        return Date.Date(0)


# Date extraction using the wsa method
def wsa_label_segment(segment):
    tokenized = LightSegmentation.split_into_tokens(segment.lower())
    clusters = StringAutomaton.extract_clusters(tokenized)
    best_run = []
    best_score = 0
    best_cluster = ""
    for cluster in clusters:
        text = tokenized[cluster[0]:cluster[1] + 1]
        run = StringAutomaton.find_best_run(text)
        score = StringAutomaton.evaluate_run(text, run)
        if score > best_score:
            best_run = run
            best_score = score
            best_cluster = text
    return StringAutomaton.get_date_from_run(best_cluster, best_run)
