import json
import random
import re
from math import ceil

import torch


from text_segmentation.LightSegmentation import split_into_sentences
from utils.Roberta import Roberta
from mpi4py import MPI

# How many random dates are generated from each occurrence
REPLACEMENTS_PER_DATE = 10
months = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november",
          "december"]
months_abbr = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
roberta = Roberta.instance()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()



def get_bert_embedding(text):
    lhs = roberta.model(**roberta.tokenizer(text, return_tensors="pt")).last_hidden_state
    lhs_vec = lhs.reshape(-1, lhs.size(2))
    sentence_embedding = torch.zeros(lhs_vec.size(1), dtype=lhs_vec.dtype)
    for i in range(lhs_vec.size(0)):
        sentence_embedding += lhs_vec[i]
    sentence_embedding /= lhs_vec.size(0)
    return sentence_embedding.detach()


def find_dates(text):
    occurrences = []
    pattern_date = re.compile("[0-3]?[0-9][.-/][0-3]?[0-9][.-/][1-9][0-9]{1,3}")
    for match in re.finditer(pattern_date, text):
        if match is None:
            break
        occurrences.append((match.start(), match.group()))
    return occurrences


def find_years(text):
    occurrences = []
    # currently limited to 3 or 4 digit years
    pattern_year = re.compile("[1-9][0-9]{2,3}")
    for match in re.finditer(pattern_year, text):
        if match is None:
            break
        occurrences.append((match.start(), match.group()))
    return occurrences


def find_months(text):
    occurrences = []
    for month in months:
        # skip may, as it usually doesn't refer to the month
        if month == "may":
            continue
        start = 0
        while True:
            occurrence = text.find(month, start)
            if occurrence == -1:
                break
            occurrences.append((occurrence, month))
            start = occurrence + 1
            if occurrence == len(text) - 1:
                break
    return occurrences


def replace_date(text, occurrence):
    separators = [".", "-", "/"]
    if "." in occurrence[1]:
        date = occurrence[1].split(".")
    elif "-" in occurrence[1]:
        date = occurrence[1].split("-")
    elif "/" in occurrence[1]:
        date = occurrence[1].split("/")
    else:
        return text
    # american MM/DD/YYYY format
    if int(date[1]) > 12:
        date[0] = random.randint(1, 12)
        date[1] = random.randint(1, 31)
        date[2] = random.randint(100, 2099)
        american_format = True
    # otherwise assume DD/MM/YYYY
    else:
        date[0] = random.randint(1, 31)
        date[1] = random.randint(1, 12)
        date[2] = random.randint(100, 2099)
        american_format = False
    left = text[:occurrence[0]]
    right = text[occurrence[0] + len(occurrence[1]):]
    separator = separators[random.randint(0, 2)]
    date_text = str(date[0]) + separator + str(date[1]) + separator + str(date[2])
    if american_format:
        norm_date = str(date[1]) + "." + str(date[0]) + "." + str(date[2])
    else:
        norm_date = str(date[0]) + "." + str(date[1]) + "." + str(date[2])
    return left + date_text + right, norm_date


def replace_year(text, occurrence):
    left = text[:occurrence[0]]
    right = text[occurrence[0] + len(occurrence[1]):]
    return left + random.randint(100, 2030) + right


def replace_month(text, occurrence):
    left = text[:occurrence[0]]
    right = text[occurrence[0] + len(occurrence[1]):]
    return left + months[random.randint(0, len(months) - 1)] + right


def synthesize_data(file):
    out_file = open("synth_data_embs.jsonl", "w", encoding="utf-8")
    lines = []
    results = []
    for line in file:
        lines.append(line)
    n = int(ceil(len(lines) / size))
    begin = rank * n
    end = (rank + 1) * n
    for line in lines[begin:end]:
        text = json.loads(line)["text"]
        for sentence in split_into_sentences(text):
            lowered = sentence.lower()
            dates = find_dates(lowered)
            if len(dates) > 0:
                for date in dates:
                    for _ in range(REPLACEMENTS_PER_DATE):
                        replacement = replace_date(lowered, date)
                        results.append(json.dumps((get_bert_embedding(replacement[0]).tolist(), replacement[1])) + "\n")
    comm.Barrier()
    gathered_results = comm.gather(results)
    if rank == 0:
        for l in gathered_results:
            for ll in l:
                out_file.write(ll)
    out_file.close()
