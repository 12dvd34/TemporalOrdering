# Loads dataset provided by https://www.kaggle.com/datasets/harishcscode/all-news-articles-from-home-page-media-house
# Some cleaning here too, some articles may be skipped if they aren't formatted well
# Returns a list of dicts with title and text of each article
def load(path):
    file = open(path, encoding="utf-8")
    file.readline()
    lines = []
    line = ""
    for l in file:
        if l.strip() == "":
            continue
        if l.split(",")[0].isdigit() and line != "":
            lines.append(line)
            line = ""
        line = line + " " + l
    content = []
    for line in lines:
        is_in_quotes = False
        splt = []
        for i, c in enumerate(line):
            if c == "\"":
                is_in_quotes = not is_in_quotes
            elif c == "," and not is_in_quotes:
                splt.append(i)
        if len(splt) != 5:
            continue
        title = line[splt[3] + 1:splt[4]].strip().replace("\"\"", "\"")
        text = line[splt[2] + 2:splt[3]].strip()[:-1].replace("\"\"", "\"")
        content.append({"title": title, "text": text})
    return content
