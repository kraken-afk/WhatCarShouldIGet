from transformers import RobertaTokenizer

tokenizer = RobertaTokenizer.from_pretrained('microsoft/graphcodebert-base')


def count_tokens(text: str) -> int:
    return len(tokenizer.tokenize(text))
