# Loads dataset provided by https://www.kaggle.com/datasets/hadasu92/cnn-articles-after-basic-cleaning
# This one has to be cleaned up a little
# Returns a list of dicts with title and text of each article
def load(path):
    file = open(path, encoding="utf-8")
    file.readline()
    content = []
    for line in file:
        if line.strip() == "":
            continue
        if not line.split(",")[0].isdigit():
            continue
        is_in_quotes = False
        splt = []
        for i, c in enumerate(line):
            if c == "\"":
                is_in_quotes = not is_in_quotes
            elif c == "," and not is_in_quotes:
                splt.append(i)
        if len(splt) != 10:
            continue
        title = line[splt[6] + 1:splt[7]].strip().replace("\"\"", "\"")
        text = line[splt[9] + 2:].strip()[:-1].replace("\"\"", "\"")
        content.append({"title": title, "text": text})
    file.close()
    return content
