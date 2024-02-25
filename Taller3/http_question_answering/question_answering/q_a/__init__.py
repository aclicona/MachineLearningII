import pickle
import os
from sentence_transformers import SentenceTransformer


def split_text_by_paragraph(filename):
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


module_dir = os.path.dirname(__file__)  # get current directory
EMBEDDING_MODEL = SentenceTransformer("avsolatorio/GIST-Embedding-v0")
TEXT_BASE = split_text_by_paragraph(f'{module_dir}/ml_models/texto_base.txt')
