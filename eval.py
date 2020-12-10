import numpy as np
import math
from pprint import pprint

def eval_scoring_fn(vocab, active_vocabs, questions):
  correct = []
  x = []
  for a, q in zip(active_vocabs, questions):
    for i, (score, word_id) in enumerate(a.scored_word_indices):
      if vocab[word_id] == q.answer:
        x.append((i, score, a.scored_word_indices[0][0], vocab[word_id]))
        correct.append(i)
        break
  pprint(sorted(x))
  print('sorted:', sorted(correct))
  print('mean:', np.mean(correct))  
  print('median:', sorted(correct)[len(correct) // 2])  


def accuracy(vocab, questions, answers):
  good = 0
  for q, a in zip(questions, answers):
    if q.answer == vocab[a]:
      good += 1
  print('%d out of %d' % (good, len(questions)))
  return 1.0 * good / len(questions)



def score(vocab, scoring_fn, answers, questions):
  words = [vocab[x] for x in answers]
  r = 0
  best = 0
  for a, q in zip(words, questions):
    r += scoring_fn([a], q)[0]
    best += scoring_fn([q.answer], q)[0]

  print('Candidate answers score:', r)
  print('True answers score:', best)
