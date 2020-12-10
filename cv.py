import numpy as np
import cv2
import operator
import numpy as np
import copy


def pre_process_image(img, skip_dilate=False):
  """Uses a blurring function, adaptive thresholding and dilation to expose the main features of an image."""

  # Gaussian blur with a kernal size (height, width) of 9.
  # Note that kernal sizes must be positive and odd and the kernel must be square.
  proc = cv2.GaussianBlur(img.copy(), (9, 9), 0)

  # Adaptive threshold using 11 nearest neighbour pixels
  proc = cv2.adaptiveThreshold(proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

  # Invert colours, so gridlines have non-zero pixel values.
  # Necessary to dilate the image, otherwise will look like erosion instead.
  proc = cv2.bitwise_not(proc, proc)

  if not skip_dilate:
    # Dilate the image to increase the size of the grid lines.
    kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]],np.uint8)
    proc = cv2.dilate(proc, kernel)

  return proc


def find_corners_of_largest_polygon(img):
  """Finds the 4 extreme corners of the largest contour in the image."""

  contours, h = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours
  contours = sorted(contours, key=cv2.contourArea, reverse=True)  # Sort by area, descending
  polygon = contours[0]  # Largest image

  # Use of `operator.itemgetter` with `max` and `min` allows us to get the index of the point
  # Each point is an array of 1 coordinate, hence the [0] getter, then [0] or [1] used to get x and y respectively.

  # Bottom-right point has the largest (x + y) value
  # Top-left has point smallest (x + y) value
  # Bottom-left point has smallest (x - y) value
  # Top-right point has largest (x - y) value
  bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
  top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
  bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
  top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))

  # Return an array of all 4 points using the indices
  # Each point is in its own array of one coordinate
  return [polygon[top_left][0], polygon[top_right][0], polygon[bottom_right][0], polygon[bottom_left][0]]


def distance_between(p1, p2):
  """Returns the scalar distance between two points"""
  a = p2[0] - p1[0]
  b = p2[1] - p1[1]
  return np.sqrt((a ** 2) + (b ** 2))


def crop_and_warp(img, crop_rect):
  """Crops and warps a rectangular section from an image into a square of similar size."""

  # Rectangle described by top left, top right, bottom right and bottom left points
  top_left, top_right, bottom_right, bottom_left = crop_rect[0], crop_rect[1], crop_rect[2], crop_rect[3]

  # Explicitly set the data type to float32 or `getPerspectiveTransform` will throw an error
  src = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')

  # Get the longest side in the rectangle
  side = max([
    distance_between(bottom_right, top_right),
    distance_between(top_left, bottom_left),
    distance_between(bottom_right, bottom_left),
    distance_between(top_left, top_right)
  ])

  # Describe a square with side of the calculated length, this is the new perspective we want to warp to
  dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype='float32')

  # Gets the transformation matrix for skewing the image to fit a square by comparing the 4 before and after points
  m = cv2.getPerspectiveTransform(src, dst)

  # Performs the transformation on the original image
  return cv2.warpPerspective(img, m, (int(side), int(side)))


def remove_box(img, top_left, bottom_right):
  return cv2.rectangle( 
    img.copy(),
    (top_left[0], top_left[1]),
    (bottom_right[0], bottom_right[1]),
    (255, 255, 255), -1)  


MARGIN = 3


def is_cell(contour, min_side_size, max_side_size):
  area = cv2.contourArea(contour)
  _, _, w, h = cv2.boundingRect(contour)

  box_area = w * h
  if w < h:
    w, h = h, w

  if h < min_side_size:
    return False

  if w > max_side_size:
    return False

  if w > 1.75 * h:
    return False

  if box_area > 1.25 * area:
    return False 

  return True


def get_cell_params(img, min_cells=10, max_cells=30):
  contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # Find contours

  min_size = int(img.shape[0] / max_cells)
  max_size = int(img.shape[0] / min_cells)

  # Filter contours
  contours = [c for c in contours if is_cell(c, min_size, max_size)]
  areas = [cv2.contourArea(c) for c in contours]
  mean_area = np.mean(areas)
  contours = [c for c, area in zip(contours, areas)
              if (abs(area - mean_area) / mean_area) < 0.25]

  wh = np.array([np.array(cv2.boundingRect(c)[2:]) for c in contours])
  w = int(np.mean(wh[:,0]) + MARGIN)
  h = int(np.mean(wh[:,1]) + MARGIN)
  num_x_cells = int(img.shape[0] / w)
  num_y_cells = int(img.shape[0] / h)
  return num_x_cells, num_y_cells



def infer_squares(img, num_x_cells, num_y_cells):
  squares = []
  side = img.shape[:1]
  w = img.shape[0] / num_x_cells
  h = img.shape[1] / num_y_cells

  squares = []
  for i in range(num_y_cells):
    row_squares = []
    for j in range(num_x_cells):
      p1 = (j * w, i * h)  # Top left corner of a bounding box
      p2 = ((j + 1) * w, (i + 1) * h)  # Bottom right corner of bounding box
      row_squares.append((p1, p2))
    squares.append(row_squares)
  return squares



def cut_from_rect(img, rect):
  """Cuts a rectangle from an image using the top left and bottom right points."""
  return img[int(rect[0][1]):int(rect[1][1]), int(rect[0][0]):int(rect[1][0])]


def lower_right_corner(rect):
  w = rect[1][0] - rect[0][0]
  h = rect[1][1] - rect[0][1]
  return ((rect[0][0] + w / 2, rect[0][1] + h / 2), rect[1])


def infer_grid(img, squares, num_x_cells, num_y_cells):
  grid = []
  for i in range(num_y_cells):
    grid_row = []
    for j in range(num_x_cells):
      cell = cut_from_rect(img, lower_right_corner(squares[i][j]))
      grid_row.append(1 if int(np.mean(cell)) > 130 else 0)
    grid.append(grid_row)
  return grid


def infer_numbers(grid):
  new_grid = copy.deepcopy(grid)
  n = len(grid)
  m = len(grid[0])
  k = 1
  for i in range(n):
    for j in range(m):
      if not grid[i][j]:
        continue
      nothing_left = (j == 0) or (j > 0 and (grid[i][j - 1] == 0)) 
      has_right = ((j + 1) < m) and (grid[i][j + 1] == 1)
      is_horizontal = nothing_left and has_right

      nothing_up = (i == 0) or (i > 0 and (grid[i - 1][j] == 0))
      has_down = ((i + 1) < n) and (grid[i + 1][j] == 1)
      is_vertical = nothing_up and has_down

      if is_horizontal or is_vertical:
        new_grid[i][j] = -k
        k += 1
      else:
        new_grid[i][j] = grid[i][j]
  return new_grid


def get_box_and_question_images(img_path):
  original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
  processed = pre_process_image(original)

  corners = find_corners_of_largest_polygon(processed)
  box = crop_and_warp(original, corners)
  questions = remove_box(original, corners[0], corners[2])

  return box, questions


def get_grid(img):
  processed = pre_process_image(img)
  num_x_cells, num_y_cells = get_cell_params(processed)

  squares = infer_squares(img, num_x_cells, num_y_cells)
  grid = infer_grid(img, squares, num_x_cells, num_y_cells)
  return infer_numbers(grid)
