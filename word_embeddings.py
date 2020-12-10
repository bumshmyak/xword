import os
import pymorphy2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import gensim

MORPH = pymorphy2.MorphAnalyzer()

# call init(data_dir) to init the model
RUSVECTORES_MODEL = None
RUSVECTORES_MODEL_WRAPPER = None


def normalize(token, append_pos=True, only_nouns=False):
  parse = MORPH.parse(token)
  if not parse:
    return token
  token = parse[0].normal_form
  pos = str(parse[0].tag.POS)
  if append_pos:
    token = token + '_' + pos
  if only_nouns:
    return token if pos == 'NOUN' else None
  return token


def tokenize(text, normalizer_fn=None):
  text = text.translate(str.maketrans('', '', ',.)(:'))
  return [w for w in text.split() if len(w) > 1 and not w.isdigit()]
  

class EmbeddingModelWrapper:
  def __init__(self, model, emb_dim, tokenizer_fn, normalizer_fn=None):
    self.model = model
    self.emb_dim = emb_dim
    self.tokenizer_fn = tokenizer_fn
    self.normalizer_fn = normalizer_fn

  def get_token_embed(self, token, normalizer_fn=None):
    if normalizer_fn:
      token = normalizer_fn(token)
    elif self.normalizer_fn:
      token = self.normalizer_fn(token)
    if token in self.model:
      return self.model[token]
    else:
      return np.zeros([self.emb_dim], dtype=np.float32)  

  def get_sentence_embed(self, sentence):
    tokens = self.tokenizer_fn(sentence)
    embeds = [self.get_token_embed(t) for t in tokens]
    embeds = [e for e in embeds if e[0] != 0.]
    if embeds:
      r = np.mean(np.array(embeds), axis=0)
      r /= np.linalg.norm(r, ord=2)
      return r
    else:
      return np.zeros([self.emb_dim], dtype=np.float32)


def get_mean_emb_scoring_fn(model, words, q, token_normalizer_fn=None):
  word_embeds = np.array([model.get_token_embed(w, token_normalizer_fn) for w in words])
  q_emb = model.get_sentence_embed(q.text)
  q_emb = q_emb.reshape((1, len(q_emb)))
  return cosine_similarity(word_embeds, q_emb)[:,0]


def init(data_dir):
  global RUSVECTORES_MODEL
  global RUSVECTORES_MODEL_WRAPPER

  RUSVECTORES_MODEL = gensim.models.KeyedVectors.load_word2vec_format(
    os.path.join(data_dir, 'tayga_upos_skipgram_300_2_2019.bin'), binary=True)
  RUSVECTORES_MODEL.init_sims(replace=True)  

  normalizer_fn = lambda token: normalize(token, append_pos=True, only_nouns=True)
  RUSVECTORES_MODEL_WRAPPER = EmbeddingModelWrapper(
      RUSVECTORES_MODEL, RUSVECTORES_MODEL.vector_size, tokenize, normalizer_fn)

def get_rusvectores_emb_scoring_fn(words, q):
  return get_mean_emb_scoring_fn(
      RUSVECTORES_MODEL_WRAPPER, words, q, token_normalizer_fn=lambda t: t + '_NOUN')
