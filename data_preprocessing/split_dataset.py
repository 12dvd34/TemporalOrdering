# Creates train/test/dev splits
# This uses JSONL: https://jsonlines.org/
file = open("articles.json", "r", encoding="utf-8")
file_train = open("articles_train.jsonl", "w", encoding="utf-8")
file_test = open("articles_test.jsonl", "w", encoding="utf-8")
file_dev = open("articles_dev.jsonl", "w", encoding="utf-8")

for index, line in enumerate(file):
    if index % 10 == 8:
        file_test.write(line)
    elif index % 10 == 9:
        file_dev.write(line)
    else:
        file_train.write(line)

file.close()
file_train.close()
file_test.close()
file_dev.close()
