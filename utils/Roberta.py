from transformers import RobertaModel, RobertaTokenizer


class Roberta:
    _instance = None

    def __init__(self):
        raise RuntimeError("Call instance() instead")

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance.tokenizer = RobertaTokenizer.from_pretrained("roberta-base", cache_dir="./cache")
            cls._instance.model = RobertaModel.from_pretrained("roberta-base", cache_dir="./cache")
        return cls._instance
