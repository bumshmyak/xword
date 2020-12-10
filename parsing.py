import re
import numpy as np
import os
from collections import defaultdict, namedtuple

from html.parser import HTMLParser
import base

Q = base.Question


class Square:
  def __init__(self, i, j, color, letter=None, number=None):
    self.i = i
    self.j = j
    self.color = color
    self.letter = letter
    self.number = number
    
  def __repr__(self):
    return '%d %d %s %s %s' % (self.i, self.j, self.color, self.letter, self.number)

class Clue:
  def __init__(self, d, number, text=None):
    self.d = d
    self.number = number
    self.text = text
    
  def __repr__(self):
    return '%d|%d|%s' % (self.d, self.number, self.text)
  
  
class MyHTMLParser(HTMLParser):
    def __init__(self):
      super(MyHTMLParser, self).__init__()
      self.squares = []
      self.clues = []
      self.is_square = False
      self.is_square_span = False
      self.is_clue_number = False
      self.is_clue_text = False
      self.clue_d = -1
  
    def handle_starttag(self, tag, attrs):
      self.is_clue_number = False
      self.is_clue_text = False
      attrs = dict(attrs)
      if tag == 'span' and attrs.get('class') == 'ClueHeading':
        self.clue_d += 1
      elif tag == 'span' and attrs.get('class') == 'ClueNumber':
        self.is_clue_number = True
      elif tag == 'span' and attrs.get('class') == 'ClueText':
        self.is_clue_text = True
      elif tag == 'input':
        self.is_square = False
        self.is_square_span = False
      elif tag == 'div' and attrs.get('class') == 'GridSquare Let':
        self.is_square = True
        style = attrs['style']
        i = None
        j = None
        color = None

        height = None
        width = None
        for a in style.split('; '):
          kv = a.split(': ')
          if len(kv) == 1:
            continue
          k, v = kv
          if k == 'height':
            height = int(v.replace('px', ''))
          elif k == 'width':
            width = int(v.replace('px', ''))
          elif k == 'top':
            i = int(v[:-2]) // height
          elif k == 'left':
            j = int(v[:-2]) // width
          elif k == 'background-color':
            color = v
        assert i is not None
        assert j is not None
        assert color is not None
        self.squares.append(Square(i, j, color))
      elif self.is_square and tag == 'span':
        self.is_square_span = True
        
    def handle_data(self, data):
      if self.is_square_span:
        try:
          n = int(data)
          self.squares[-1].number = n
        except:
          if len(data) == 1:
            self.squares[-1].letter = data.lower() # .replace('ё', 'е')
      elif self.is_clue_number:
        n = int(data[:-2])
        self.clues.append(Clue(self.clue_d, n))
      elif self.is_clue_text:
        self.clues[-1].text = base.normalize_question(data[:-1])
        
    def handle_endtag(self, tag):
      if tag == 'span':
        self.is_square_span = False

        
def normalize_answer(w):
  return w.lower()


def extract_questions(board, squares, clues, n, m):
  clues_dict = {}
  for c in clues:
    clues_dict[(c.d, c.number)] = c.text
  questions = []
  for s in squares:
    if s.number is not None:
      assert board[s.i][s.j] != '#'
      if s.j + 1 < m and board[s.i][s.j + 1] != '#' and (s.j == 0 or board[s.i][s.j - 1] == '#'):
        w = ''
        j = s.j
        while j < m and board[s.i][j] != '#':
          w += board[s.i][j]
          j += 1
        questions.append(Q(s.i, s.j, 0, len(w), s.number, clues_dict[(0, s.number)], normalize_answer(w)))
      if s.i + 1 < n and board[s.i + 1][s.j] != '#' and (s.i == 0 or board[s.i - 1][s.j] == '#'):
        w = ''
        i = s.i
        while i < n and board[i][s.j] != '#':
          w += board[i][s.j]
          i += 1
        questions.append(Q(s.i, s.j, 1, len(w), s.number, clues_dict[(1, s.number)], normalize_answer(w)))
  return questions        


def parse(html_filename):
  parser = MyHTMLParser()
  parser.feed(open(html_filename).read())
  squares = parser.squares

  n = np.max([s.i for s in squares]) + 1
  m = np.max([s.j for s in squares]) + 1
  
  initial_board = np.array([np.array(list('#' * m)) for _ in range(n)])
  answer_board = np.array([np.array(list('#' * m)) for _ in range(n)])

  for s in squares:
    answer_board[s.i][s.j] = s.letter
    initial_board[s.i][s.j] = '.'
    
  questions = extract_questions(answer_board, squares, parser.clues, n, m)

  return questions
