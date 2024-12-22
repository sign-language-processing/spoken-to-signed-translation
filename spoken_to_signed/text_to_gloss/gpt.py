import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import List

from openai import OpenAI
from dotenv import load_dotenv

from spoken_to_signed.text_to_gloss.types import Gloss, GlossItem

SYSTEM_PROMPT = """
You are a helpful assistant, who helps glossify sentences into sign language glosses. 
Your task is to convert spoken language text into glossed sign language sentences following specific formatting rules.

Follow these guidelines:

1. **Sentence Structure**:
   - Gloss each sentence separately. Break down long sentences into distinct, meaningful glosses for clarity.
   - Respond with a list of glossed sentences, each reflecting the structure of the original spoken sentence, using glosses and corresponding words.
   - Prefer SOV (Subject-Object-Verb) word order for glossing.

2. **Glossing Rules**:
   - Translate words into glosses in uppercase.
   - Retain the original spoken words alongside their glosses. 
   - Place a slash `/` between the gloss and the original word to denote a direct translation. For example, "HELLO/Hello" or "NAME/name."

3. **Mouthing Notation**:
   - For sign languages with mouthing (such as German Sign Language - `gsg`), include the mouthing symbol `⌘` with the gloss. 
   - Use open brackets to indicate the gloss that matches the mouth movement, for example: `⌘schön(SCHÖN/schöne)`.

4. **Named Entities**:
   - For proper nouns or named entities, use the `%` symbol in place of glossing. This denotes spelling out the entity instead of providing a gloss, for instance, `%(Inigo Montoya)`.

Use these rules and examples to produce accurate and readable glosses for each sentence provided.
""".strip()


@lru_cache(maxsize=1)
def get_openai_client():
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY", None)
    return OpenAI(api_key=api_key)


@lru_cache(maxsize=1)
def few_shots():
    data_path = Path(__file__).parent / "few_shots.json"
    with open(data_path, 'r', encoding="utf-8") as file:
        data = json.load(file)

    messages = []
    for entry in data:
        messages.append({
            "role": "user",
            "content": json.dumps({
                "spoken_language": entry['spoken_language'],
                "signed_language": entry['signed_language'],
                "text": entry['text'],
            })
        })
        messages.append({"role": "assistant", "content": json.dumps(entry['sentences'])})

    return messages


def sentence_to_glosses(sentence: str) -> GlossItem:
    for item in sentence.split(" "):
        regex_with_mouthing = r"⌘(.*?)\((.*?)\)"
        if match := re.match(regex_with_mouthing, item):
            mouthing = match.group(1)
            content = match.group(2)
        else:
            mouthing = None
            content = item
        for sub_item in content.split(" "):
            if "/" in sub_item:
                sub_item_gloss, sub_item_word = sub_item.split("/")
            else:
                sub_item_gloss = sub_item_word = sub_item
            yield sub_item_word, sub_item_gloss


def text_to_gloss(text: str, language: str, signed_language: str, **kwargs) -> List[Gloss]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + few_shots() + [{
        "role": "user",
        "content": json.dumps({
            "spoken_language": language,
            "signed_language": signed_language,
            "text": text,
        })}
    ]

    response = get_openai_client().chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        seed=42,
        messages=messages,
        max_tokens=500
    )

    prediction = response.choices[0].message.content
    print(prediction)
    sentences = json.loads(prediction)
    return [list(sentence_to_glosses(sentence)) for sentence in sentences]

if __name__ == '__main__':
    text = "Kleine kinder essen pizza."
    language = "de"
    signed_language = "sgg"
    print(text_to_gloss(text, language, signed_language))
