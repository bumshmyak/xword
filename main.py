#!/usr/bin/env python

import sys
import os
import pickle
import copy
import numpy as np
import subprocess

from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from pprint import pprint

import ocr
import parsing
import base
import qdb
import word_embeddings
import sentence_embeddings
import scoring


def get_db_bert_similarity_scoring_fn(words, q):
  return scoring.get_db_similarity_scoring_fn(
      QUESTION2ID, DB_ANSWERS, BERT_DB_SIMILARITY, words, q)


def get_scoring_fn(words, q):
  word_scores = word_embeddings.get_rusvectores_emb_scoring_fn(words, q)
  db_scores = get_db_bert_similarity_scoring_fn(words, q)
  return scoring.demote_question_words(words, q, np.fmax(word_scores, db_scores))  


class Scorer:
  def __init__(self, score_fn):
    self.scores = []
    for q in QUESTIONS:
      words = [VOCAB[i] for i in VOCAB_INDICES_BY_LENGTH[q.length]]
      self.scores.append(score_fn(words, q))

  def score(self, q_index, w_index):
    return self.scores[q_index][VOCAB_TO_VOCAB_BY_LENGTH[w_index]]


class ActiveVocab:
  def __init__(self, q_index, scored_word_indices):
    self.q_index = q_index
    self.scored_word_indices = scored_word_indices
    self.scored_word_indices_by_pos = {}
    for _, pos, _ in GRAPH[self.q_index]:
      if pos not in self.scored_word_indices_by_pos:
        self.scored_word_indices_by_pos[pos] = defaultdict(list)
      for score, word_index in self.scored_word_indices:
        self.scored_word_indices_by_pos[pos][VOCAB[word_index][pos]].append((score, word_index))
    
  def restrict(self, pos, char):
    return ActiveVocab(self.q_index, self.scored_word_indices_by_pos[pos][char])

  def max_restricted_score(self, pos, char):
    t = self.scored_word_indices_by_pos[pos][char]
    if not t:
      return None
    else:
      return t[0][0]
  
  def max_score(self):
    return self.scored_word_indices[0][0]
  
  @classmethod
  def create(self, scorer, q_index):
    length = QUESTIONS[q_index].length
    scored_word_indices = [(scorer.score(q_index, w_index), w_index)
                           for w_index in VOCAB_INDICES_BY_LENGTH[length]]
    scored_word_indices = sorted(scored_word_indices, reverse=True)
    return ActiveVocab(q_index, scored_word_indices)



def export(prefix, vocab, questions, active_vocabs, graph):
  if not os.path.exists(prefix):
    os.makedirs(prefix)
  
  open(os.path.join(prefix, 'vocab.txt'), 'w').write('\n'.join(vocab))

  open(os.path.join(prefix, 'answers.txt'), 'w').write('\n'.join(
      [str(q.length) for q in questions]))

  with open(os.path.join(prefix, 'scores.txt'), 'w') as out:
    for v in active_vocabs:
      for score, word_index in v.scored_word_indices:
        out.write('%s %s %s\n' % (v.q_index, word_index, score))

  with open(os.path.join(prefix, 'graph.txt'), 'w') as out:
    for src, edges in graph.items():
      for (dst, src_pos, dst_pos) in edges:
        out.write('%d %d %d %d\n' % (src, dst, src_pos, dst_pos))


def read_answers(filepath):
  answers = []
  with open(filepath) as f:
    for line in f.readlines():
      answers.append(line.strip())
  return answers


def show_questions_grid(questions):
  n = 0
  m = 0
  for q in questions:
    if q.d == 1:
      n = max(n, q.i + q.length)
    else:
      m = max(m, q.j + q.length)
  
  board = [['#'] * m for _ in range(n)]
  for q in questions:
    i = q.i
    j = q.j
    for k in range(q.length):
      board[i][j] = q.answer[k]
      k += 1
      if q.d == 0:
        j += 1
      else:
        i += 1
  return '\n'.join([' '.join(row) for row in board])



############################ MAIN ############################

assert len(sys.argv) == 3
DATA_DIR = sys.argv[1]
img_path = sys.argv[2]

print('Initializing models...')
word_embeddings.init(DATA_DIR)

# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '~/key.json'

print('Recognizing image...')
QUESTIONS = ocr.recognize(img_path)
QUESTION2ID = {q.text: i for i, q in enumerate(QUESTIONS)}  

print('Loading data...')
QDF = qdb.get_questions_db(os.path.join(DATA_DIR, 'baza.csv'))

GRAPH = base.get_intersection_graph(QUESTIONS)

VOCAB = list(map(lambda x: x.strip().lower(),
                 open(os.path.join(DATA_DIR, 'hagen_nouns.txt')).readlines()))
VOCAB = [w for w in VOCAB if '-' not in w]
VOCAB = list(set(VOCAB) | set(QDF['answer']))
WORD2ID = {w: i for i, w in enumerate(VOCAB)}

print('# Questions:', len(QUESTIONS))
print('# Intersections:', sum(map(len, GRAPH.values())) / 2)
print('# Words:', len(VOCAB))
print('# Questions DB:', len(QDF))


print('Calculating BERT embeddings...')
QUESTION_BERT_EMBEDS = sentence_embeddings.get_bert_embeds([q.text for q in QUESTIONS])
QDF_EMBEDS = pickle.load(open(os.path.join(DATA_DIR, 'qdf_embeds.pkl'), 'rb'))
DB_BERT_EMBEDS = np.array([QDF_EMBEDS[x] for x in QDF['question']])
DB_ANSWERS = list(QDF['answer'])
BERT_DB_SIMILARITY = cosine_similarity(QUESTION_BERT_EMBEDS, DB_BERT_EMBEDS)  


print('Preparing answers...')
VOCAB_INDICES_BY_LENGTH = defaultdict(list)
VOCAB_TO_VOCAB_BY_LENGTH = []
for i, w in enumerate(VOCAB):
  VOCAB_TO_VOCAB_BY_LENGTH.append(len(VOCAB_INDICES_BY_LENGTH[len(w)]))
  VOCAB_INDICES_BY_LENGTH[len(w)].append(i)

scorer = Scorer(get_scoring_fn)
ACTIVE_VOCABS = [ActiveVocab.create(scorer, i) for i in range(len(QUESTIONS))]

print('Solving...')
export(os.path.join(DATA_DIR, 'f'), VOCAB, QUESTIONS, ACTIVE_VOCABS, GRAPH)
args = (
    './solver',
    f'{DATA_DIR}/f/vocab.txt',
    f'{DATA_DIR}/f/answers.txt',
    f'{DATA_DIR}/f/scores.txt',
    f'{DATA_DIR}/f/graph.txt',
    f'{DATA_DIR}/f/res.txt'
)
popen = subprocess.Popen(args, stdout=subprocess.PIPE)
popen.wait()
output = popen.stdout.read()

print('Result...')
answers = read_answers(os.path.join(DATA_DIR, 'f', 'res.txt'))
for q, a in zip(QUESTIONS, answers):
  q.answer = a

questions = copy.deepcopy(QUESTIONS)
questions = sorted(questions, key=lambda q: (q.d, q.number))
h_answers = [(q.number, q.answer) for q in questions if q.d == 0]
v_answers = [(q.number, q.answer) for q in questions if q.d == 1]
print('Horizontal:', h_answers)
print('Vertical:', v_answers)

print(show_questions_grid(QUESTIONS))  

