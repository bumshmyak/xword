import numpy as np
from word_embeddings import normalize, tokenize


def get_db_similarity_scoring_fn(question2id, db_answers, similarity, words, q):
  q_id = question2id[q.text]
  answer2score = {}
  for a, s in zip(db_answers, list(similarity[q_id,:])):
    answer2score[a] = max(answer2score.get(a, 0.), s)
  return [answer2score.get(w, 0.0) for w in words]


def normalizer_fn(token):
  return normalize(token, append_pos=True, only_nouns=True)


def demote_question_words(words, q, scores):
  question_tokens = set([normalizer_fn(t) for t in tokenize(q.text)])
  is_in_question = np.array([((t + '_NOUN') in question_tokens) for t in words])
  return np.where(is_in_question, 0., scores)

