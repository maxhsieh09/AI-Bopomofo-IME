from pathlib import Path
import re
import json
from tqdm import tqdm

import opencc

def clean_wiki_text(text):
    # Remove [[Image:...]] and [[File:...]]
    text = re.sub(r'\[\[(Image|File):.*?\]\]', '', text)
    # Remove [[Category:...]]
    text = re.sub(r'\[\[Category:.*?\]\]', '', text)
    # Remove template style / html
    text = re.sub(r'&lt;.*?&gt;', '', text)
    # Remove links but keep the display text
    text = re.sub(r'\[\[([^|\]]*\|)?([^\]]+)\]\]', r'\2', text)
    # Remove title translation brackets
    text = re.sub(r'（）', '', text)
    return text


def preprocess_wiki_data(file_path):
    articles = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                articles.append(json.loads(line))

    texts = []
    converter = opencc.OpenCC("s2t.json")
    for article in articles:
        text = article["text"]
        text = clean_wiki_text(text)
        text = text.replace("\n", "")
        text = converter.convert(text)
        if len(text) > 6:
            texts.append(text)

    return texts

texts = []
for subdir in Path("wiki_extracted/").iterdir():
    if subdir.is_dir():
        for file_path in tqdm(subdir.iterdir(), total=100):
            if file_path.name.startswith("wiki_"):
                texts.extend(preprocess_wiki_data(file_path))

with open("preprocessed_wiki.txt", "w", encoding="utf-8") as f:
    for text in texts:
        f.write(text + "\n")
