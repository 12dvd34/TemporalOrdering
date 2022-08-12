import csv


# Loads the dataset provided by https://www.kaggle.com/datasets/hgultekin/bbcnewsarchive
# It's a CSV file, but with tabs, so it has to be read as a TSV
# Returns a list of dicts with title and text of each article
def load(path):
    tsv_file = open(path, encoding="utf-8")
    reader = csv.reader(tsv_file, dialect="excel-tab")
    content = []
    tsv_file.readline()
    for line in reader:
        content.append({"title": line[2], "text": line[3]})
    tsv_file.close()
    return content
