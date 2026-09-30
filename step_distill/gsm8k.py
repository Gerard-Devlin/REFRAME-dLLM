"""GSM8K text handling for the retained v2 distillation experiment."""

from decimal import Decimal, InvalidOperation
import re


def extract_answer(text, gold=False):
    if gold:
        text = text.split("####")[-1]
    else:
        explicit = re.findall(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", text)
        boxed = re.findall(r"\\boxed\{\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*\}", text)
        values = explicit or boxed or re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
        if not values:
            return None
        text = values[-1]
    try:
        return str(Decimal(text.strip().replace(",", "")).normalize())
    except InvalidOperation:
        return None


def prompt_ids(tokenizer, question):
    return tokenizer.apply_chat_template([
        {"role": "user", "content": question + "\nExplain your reasoning and end with #### followed by the final number."}
    ], tokenize=True, add_generation_prompt=True)
