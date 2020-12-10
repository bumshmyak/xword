from google.cloud import vision
import re
from collections import namedtuple
import functools
import math
import cv2

import base
import cv


def ocr_image(img_path):
  client = vision.ImageAnnotatorClient()
  image = vision.Image()

  with open(img_path, "rb") as image_file:
    image.content = image_file.read()

  response = client.text_detection(image=image)
  return response


Box = namedtuple('Box', ['top_left', 'bottom_right', 'text'])


def _find_thresholds(boxes, margin=0):
  xs = []
  for b in boxes:
    xs.append((b.top_left.x, 0))
    xs.append((b.bottom_right.x, 1))
  xs = sorted(xs)
  
  thresholds = [0]
  b = 0
  last = None
  for (x, t) in xs:
    if b == 0 and (last is not None) and (x - last) > margin:
      thresholds.append((x + last) / 2)
      last = None
    b += (1 if (t == 0) else -1)
    if b == 0:
      last = x
  thresholds.append(xs[-1][0] + 1)
      
  return thresholds


def _infer_columns(boxes):
  thresholds = _find_thresholds(boxes)
  columns = []
  for b in boxes:
    x = b.top_left.x
    for i, (first, second) in enumerate(zip(thresholds, thresholds[1:])):
      if x > first and x <= second:
        columns.append(i)
        break
  return columns, len(thresholds) - 2


def _cmp(is_less):
  return -1 if is_less else 1


def _box_comparator(first, second, margin=20):
  first_column, first = first
  second_column, second = second

  if first_column != second_column:
    return _cmp(first_column < second_column)

  first = first.top_left
  second = second.top_left

  if math.fabs(first.y - second.y) > margin:
    return _cmp(first.y < second.y)
  else:
    return _cmp(first.x < second.x)


def _fix_boxes_ordering(boxes):
  columns, num_columns = _infer_columns(boxes)
  return [b for c, b in sorted(list(zip(columns, boxes)),
                               key=functools.cmp_to_key(_box_comparator))]


def extract_boxes(response):
  boxes = []
  for text in response.text_annotations[1:]:
    top_left = text.bounding_poly.vertices[0]
    bottom_right = text.bounding_poly.vertices[2]
    norm_text = text.description.replace('|', '').strip()
    if norm_text:
      boxes.append(Box(top_left, bottom_right, norm_text))
  return boxes



def _normalize_line(line):
  return line.replace('Т.', '1.').strip().lower()


def extract_questions(boxes):
  res = []
  current = ''
  current_index = None
  horizontal = None
  vertical = None
  all_text = ''

  for b in boxes:
    line = _normalize_line(b.text)
    all_text += ' ' + line

    if all_text.find('по вертикали') != -1:
      if current_index is not None:
        current = current.replace('по вертикали', '')
        if current.endswith(' по'):
          current = current[:-3]
        res.append((current_index, base.normalize_question(current)))

      horizontal = res
      res = []
      current = ''
      current_index = None
      all_text = ''

    m = re.match(r'\d+\.', line)
    if m is not None:
      # Start question
      if current_index is not None:
        res.append((current_index, base.normalize_question(current)))     

      current_index = int(line[:m.span()[1] - 1])
      current = line[m.span()[1] + 1:]
      continue
    
    if current_index is not None:
      current += ' ' + line
    
  if current:
    res.append((current_index, base.normalize_question(current)))
  vertical = res
  return horizontal, vertical


def _recognize(h_questions, v_questions, grid):
  num2pos = {}
  for i in range(len(grid)):
    for j in range(len(grid[i])):
      if grid[i][j] < 0:
        num2pos[-grid[i][j]] = (i, j)
  
  questions = []
  for k, text in h_questions:
    (i, j) = num2pos[k]
    length = 1
    while ((j + length) < len(grid[i])) and grid[i][j + length] != 0:
      length += 1
    questions.append(base.Question(i, j, 0, length, k, text))

  for k, text in v_questions:
    (i, j) = num2pos[k]
    length = 1
    while ((i + length) < len(grid)) and grid[i + length][j] != 0:
      length += 1
    questions.append(base.Question(i, j, 1, length, k, text))

  return questions


def recognize(img_path):
  box_img, questions_img = cv.get_box_and_question_images(img_path)
  cv2.imwrite('questions.jpg', questions_img)  
  response = ocr_image('questions.jpg')
  boxes = extract_boxes(response)
  boxes = _fix_boxes_ordering(boxes)
  h_questions, v_questions = extract_questions(boxes)
  grid = cv.get_grid(box_img)
  return _recognize(h_questions, v_questions, grid)

