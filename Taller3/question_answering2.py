from sentence_transformers import SentenceTransformer, util
import numpy as np


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'


def split_text_by_dot(filename):
    """
    Opens a text file and splits its content by dot mark.

    Args:
        filename: The name of the text file to open.

    Returns:
        A list of strings, where each string is a sentence from the text file
        ending with a dot.
    """
    with open(filename, 'r') as file:
        text_ = file.read()
    sentences = text_.split("\n")
    return [sentence for sentence in sentences if sentence.strip()]


model = SentenceTransformer("Salesforce/SFR-Embedding-Mistral")
task = 'Given a web search query, retrieve relevant passages that answer the query'
text = split_text_by_dot("texto_base.txt")
queries = [
    get_detailed_instruct(task, '¿por qué fue aclamada Coraima Torres?')
]
embeddings = model.encode(queries + text)
scores = util.cos_sim(embeddings[:1], embeddings[1:]) * 100
order = np.argsort(scores.tolist()[0])[::-1].astype(int)
text = np.array(text)
print(text[order[:3]])
