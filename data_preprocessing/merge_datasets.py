import load_bbc
import load_cnn
import load_na
import os
import json

# Loads datasets and combines them into one file
path_output_file = "articles.json"
# Add paths to datasets here
path_bbc = "add local path"
path_cnn1 = "add local path"
path_cnn2 = "add local path"
# Use the path for the root folder of this dataset
path_na = "add local path"
na_datasets = ["cnn", "foxnews", "huffingtonpost", "nytimes", "reuters"]

filenames = os.listdir(path_na)
na_folders = []
for filename in filenames:
    if os.path.isdir(os.path.join(os.path.abspath(path_na), filename)):
        na_folders.append(filename)
na_folders.sort()

articles = []

for folder in na_folders:
    for dataset in na_datasets:
        path = path_na + folder + "/" + dataset + ".csv"
        articles.extend(load_na.load(path))

# Add new datasets to load here
# Or remove those not needed
articles.extend(load_bbc.load(path_bbc))
articles.extend(load_cnn.load(path_cnn1))
articles.extend(load_cnn.load(path_cnn2))
output_file = open(path_output_file, "w", encoding="utf-8")
for article in articles:
    output_file.write(json.dumps(article, ensure_ascii=False) + "\n")
output_file.close()