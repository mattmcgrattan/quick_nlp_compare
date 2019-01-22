import spacy
from flair.data import Sentence
from flair.models import SequenceTagger
import json
from dictdiffer import diff


def spacy_ner(text, nlp):
    """
    Tag with Spacy.io

    :param text: source text to tag
    :param nlp: Spacy.io initialised with language model
    :return: list of tuples (text, start, end, entity label)
    """
    doc = nlp(text)
    ents = [(e.text, e.start_char, e.end_char, e.label_) for e in doc.ents]
    return ents


def flair_ner(text, tagger):
    """
    Tag with Flair

    :param text: source text to tag
    :param tagger: Flair initialised with tagging model
    :return: list of tuples (text, start, end, entity label)
    """
    sentence = Sentence(text, use_tokenizer=True)
    tagger.predict(sentence)
    s = sentence.to_dict(tag_type="ner")
    ents = [(e["text"], e["start_pos"], e["end_pos"], e["type"]) for e in s["entities"]]
    return ents


def combined_ner(source_text, spacy_nlp, flair_tagger):
    s = spacy_ner(source_text, spacy_nlp)
    f = flair_ner(source_text, flair_tagger)
    d = {"text": source_text, "spacy_entities": spacy_ner(source_text, spacy_nlp),
         "flair_entities": flair_ner(source_text, flair_tagger), "diff": diff(s, f)}
    print(len(d["spacy_entities"]))
    print(len(d["flair_entities"]))
    return d


if __name__ == "__main__":
    n = spacy.load('en_core_web_lg')
    t = SequenceTagger.load('ner-ontonotes')
    with open("george_washington.txt", "r") as f:
        washington = f.read()
    d1 = "San Francisco is a city in California. John Smith lives there with his wife Mary and " \
        "their two dogs, Biff and Chip."
    d2 = "The Navajo are a Native American tribe found throughout the American West, especially " \
         "in New Mexico and Arizona."
    # print(json.dumps(combined_ner(source_text=d1, spacy_nlp=n, flair_tagger=t), indent=4))
    # print(json.dumps(combined_ner(source_text=d2, spacy_nlp=n, flair_tagger=t), indent=4))
    print(json.dumps(combined_ner(source_text=washington, spacy_nlp=n, flair_tagger=t), indent=4))

