import random
from segment_labeling.Date import Date
from text_segmentation.LightSegmentation import split_into_tokens


months = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november",
          "december"]
months_abbr = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
eras = ["bc", "bce", "ad", "ce"]
categories = ["day", "month", "year", "era", "other"]


# Finds the parts of the text that potentially contain dates
def extract_clusters(text):
    clusters = []
    for i in range(len(text)):
        if text[i].isdigit():
            clusters.append((i, i))
    skip = False
    while True:
        new_clusters = []
        merges = 0
        for i in range(len(clusters) - 1):
            if skip:
                skip = False
                continue
            if clusters[i + 1][0] - clusters[i][1] <= 3:
                new_clusters.append((clusters[i][0], clusters[i + 1][1]))
                merges += 1
                skip = True
            else:
                new_clusters.append(clusters[i])
                if i == len(clusters) - 2:
                    new_clusters.append(clusters[i + 1])
        if merges == 0:
            expanded_clusters = []
            for cluster in clusters:
                expanded_clusters.append((max(0, cluster[0] - 2), min(len(text) - 1, cluster[1] + 2)))
            return expanded_clusters
        else:
            clusters = new_clusters


# In a wsa we would calculate the transition mapping depending on two states (q and q') and a token
# But in this case the result would be the same for any q', so we simplify it
def transition_weight(q, token):
    # token is a number
    if token.isdigit():
        # potential day or month or year or other
        if 0 < int(token) < 13:
            if q == "day":
                return 0.25
            elif q == "month":
                return 0.5
            elif q == "year":
                return 0.01
            elif q == "era":
                return 0
            else:
                return 0.2
        # potential day or year or other
        elif 12 < int(token) < 32:
            if q == "day":
                return 0.75
            elif q == "month":
                return 0
            elif q == "year":
                return 0.01
            elif q == "era":
                return 0
            else:
                return 0.2
        # potential other
        elif int(token) < 0:
            if q == "day":
                return 0
            elif q == "month":
                return 0
            elif q == "year":
                return 0
            elif q == "era":
                return 0
            else:
                return 1
        # potential year or other
        else:
            if q == "day":
                return 0
            elif q == "month":
                return 0
            elif q == "year":
                return 0.5
            elif q == "era":
                return 0
            else:
                return 0.5
    # token is text
    else:
        # potential month or other
        if token in months or token in months_abbr:
            # may may be a month
            if token == "may":
                if q == "day":
                    return 0
                elif q == "month":
                    return 0.25
                elif q == "year":
                    return 0
                elif q == "era":
                    return 0
                else:
                    return 0.5
            # not may -> definitely a month
            else:
                if q == "day":
                    return 0
                elif q == "month":
                    return 1
                elif q == "year":
                    return 0
                elif q == "era":
                    return 0
                else:
                    return 0
        # potential era or other
        elif token in eras:
            if q == "day":
                return 0
            elif q == "month":
                return 0
            elif q == "year":
                return 0
            elif q == "era":
                return 1
            else:
                return 0.01
        # potential other
        else:
            if q == "day":
                return 0
            elif q == "month":
                return 0
            elif q == "year":
                return 0
            elif q == "era":
                return 0
            else:
                return 0.5


# Generate random initial run
def generate_run(text):
    run = []
    local_cats = categories.copy()
    for _ in range(len(text)):
        if len(local_cats) > 1:
            run.append(local_cats.pop(random.randint(0, len(local_cats) - 1)))
        else:
            run.append("other")
    return run


# User a heuristic to generate an initial run
# This will already solve some date occurrences
def init_run(text):
    assigned_cats = []
    run = []
    for i in range(len(text)):
        if text[i].isdigit():
            if 0 < int(text[i]) < 13 and "month" not in assigned_cats:
                run.append("month")
                assigned_cats.append("month")
            elif 0 < int(text[i]) < 32 and "day" not in assigned_cats:
                run.append("day")
                assigned_cats.append("day")
            elif 0 <= int(text[i]) and "year" not in assigned_cats:
                run.append("year")
                assigned_cats.append("year")
            else:
                run.append("other")
        elif text[i] in months or text[i] in months_abbr and "month" not in assigned_cats:
            run.append("month")
            assigned_cats.append("month")
        elif text[i] in eras and "era" not in assigned_cats:
            run.append("era")
            assigned_cats.append("era")
        else:
            run.append("other")
    return run


# Switch two labels randomly
def permute_run(run, swap_1=-1, swap_2=-1):
    new_run = run.copy()
    if swap_1 == -1:
        swap_1 = random.randint(0, len(run) - 1)
    if swap_2 == -1:
        swap_2 = random.randint(0, len(run) - 1)
        while swap_1 == swap_2:
            swap_2 = random.randint(0, len(run) - 1)
    new_run[swap_1] = run[swap_2]
    new_run[swap_2] = run[swap_1]
    return new_run


# Calculate score for a run on a text
def evaluate_run(text, run):
    score = 1
    for i in range(len(text)):
        score *= transition_weight(run[i], text[i])
    return score


# Human-readable representation of a run
def run_to_string(text, run):
    string = ""
    for i in range(len(text)):
        string += text[i] + "(" + run[i][0] + ") "
    string += "- " + str(evaluate_run(text, run))
    return string


# Applies random changes to an initial run to find the best one
def find_best_run(text):
    run = init_run(text)
    best_score = evaluate_run(text, run)
    best_run = run.copy()
    for _ in range(1000):
        run = permute_run(run)
        score = evaluate_run(text, run)
        if score > best_score:
            best_score = score
            best_run = run.copy()
    return best_run


# Extracts date from text as defined by the run
def get_date_from_run(text, run):
    day = 1
    month = 1
    year = 1
    for i in range(len(text)):
        if run[i] == "day":
            day = int(text[i])
        elif run[i] == "month":
            if text[i].isdigit():
                month = int(text[i])
            else:
                if text[i] in months:
                    for j in range(len(months)):
                        if text[i] == months[j]:
                            month = j + 1
                            break
                elif text[i] in months_abbr:
                    for j in range(len(months_abbr)):
                        if text[i] == months_abbr[j]:
                            month = j + 1
                            break
        elif run[i] == "year":
            year = int(text[i])
    return Date(year, month, day)