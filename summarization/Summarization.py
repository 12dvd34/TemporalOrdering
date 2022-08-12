from transformers import PegasusForConditionalGeneration, PegasusTokenizer

tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum", cache_dir="./cache")
model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum", cache_dir="./cache")


def summarize(text):
    batch = tokenizer(text, truncation=True, padding="longest", return_tensors="pt")
    translated = model.generate(**batch)
    result = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return result[0]
