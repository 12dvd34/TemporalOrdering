# Data Preprocessing

This folder contains scripts for loading data from various datasets.

### Loading Datasets

Scripts for loading from the following datasets are already provided:
- BBC News Archive (https://www.kaggle.com/datasets/hgultekin/bbcnewsarchive): `load_bbc.py`
- CNN News Articles from 2011 to 2022 (https://www.kaggle.com/datasets/hadasu92/cnn-articles-after-basic-cleaning): 
`load_cnn.py`
- News Articles (https://www.kaggle.com/datasets/harishcscode/all-news-articles-from-home-page-media-house): 
`load_na.py`

New datasets can be integrated by:
- providing a loading function returning a list of dictionaries with two keys: `title` and `text`
- adding the path to the dataset file to `merge_datasets.py`
- loading the dataset by extending the list of all articles in `merge_datasets.py` 
(see comments in the file for details)

### Merging Datasets

Add the paths to the dataset files (usually CSVs) to `merge_datasets.py`, then run the script.

### Generating Splits

Run `split_dataset.py` to generate train, test and dev splits in a 80-10-10 ratio. If the output file in 
`merge_datasets.py` has not been changed, nothing has to be done here. This will output files in the JSONL format
(https://jsonlines.org/).