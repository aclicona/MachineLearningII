import numpy as np
from sentence_transformers import util


def get_most_related_paragraphs(model, text, question):
    question = [question]
    full_text = question + text
    embeddings = model.encode(full_text, convert_to_tensor=True)
    scores = util.cos_sim(embeddings[:1], embeddings[1:]) * 100
    order = np.argsort(scores.tolist()[0])[::-1].astype(int)
    text = np.array(text)
    return text[order[:3]]
