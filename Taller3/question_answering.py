import numpy as np
from sentence_transformers import SentenceTransformer, util
import pickle


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


def save_model(model_, model_name, path='http_question_answering/question_answering/q_a/ml_models'):
    pickle.dump(model_, open(f'{path}/{model_name}.pkl', 'wb'))


text = split_text_by_dot("texto_base.txt")
question = ['¿por qué fue aclamada Coraima Torres?']
full_text = question + text

#
model = SentenceTransformer("avsolatorio/GIST-Embedding-v0")

# embeddings = model.encode(full_text, convert_to_tensor=True)
# scores = util.cos_sim(embeddings[:1], embeddings[1:]) * 100
# order = np.argsort(scores.tolist()[0])[::-1].astype(int)
text = np.array(text)
save_model(model, model_name='GIST-Embedding')
