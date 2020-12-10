from collections import namedtuple, defaultdict

STRINGS_TO_REMOVE = [',', '.', '"', '- ', '– ', '!']


def normalize_question(text):
  text = text.lower()
  for s in STRINGS_TO_REMOVE:
    text = text.replace(s, '')
  return text.strip()


class Question:
  def __init__(self, i, j, d, length, number=None, text=None, answer=None):
    self.i = i
    self.j = j
    self.d = d  # direction, 0 - horizontal, 1 - vertical
    self.length = length
    self.number = number
    self.text = text
    self.answer = answer
    
  def __repr__(self):
    return '%d|%d|%d|%d|%d|%s|%s' % (
      self.i, self.j, self.d, self.length, self.number, self.text, self.answer)


Edge = namedtuple('Edge', 'q_index first_pos second_pos')


def get_intersection_graph(questions):
  
  def _expand(q):
    res = [(q.i, q.j)]
    k = 1
    while k < q.length:
      i, j  = res[-1]
      if q.d == 0:
        j += 1
      else:
        i += 1
      res.append((i, j))
      k += 1
    return res
  
  g = defaultdict(list)
  
  for i in range(len(questions)):
    for j in range(len(questions)):
      if i < j and questions[i].d != questions[j].d:
        first = _expand(questions[i])
        second = _expand(questions[j])
        x = set(first) & set(second) 
        if x:
          assert(len(x) == 1)
          x = list(x)[0]
          g[i].append(Edge(j, first.index(x), second.index(x)))
          g[j].append(Edge(i, second.index(x), first.index(x)))
  
  return g
