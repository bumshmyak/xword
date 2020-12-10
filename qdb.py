import pandas as pd

def get_questions_db(filepath):
  df = pd.read_csv(filepath)
  df['answer'] = df['answer'].str.lower()
  df['question'] = df['question'].str.lower()
  df['question'] = df['question'].str.replace(',', '')
  df['question'] = df['question'].str.strip('?')
  df['num_answer_words'] = df.answer.map(lambda x: len(x.split()))
  df = df[df['num_answer_words'] == 1]
  df = df[df['num_answer_words'] == 1]
  df = df[df.answer.map(len) >= 3]
  df = df[df.answer.map(len) < 32]
  df = df[df.answer.str.match(r'[А-я]+')]
  df = df[df.answer.map(lambda x: '-' not in x)]
  return df
