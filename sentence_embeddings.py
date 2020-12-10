import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel


if torch.cuda.is_available():
  from torch.cuda import FloatTensor, LongTensor
  DEVICE = torch.device('cuda')
else:
  from torch import FloatTensor, LongTensor
  DEVICE = torch.device('cpu')


#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def to_device(encoded_input):
  return {k: v.to(DEVICE) for k, v in encoded_input.items()}


#Load AutoModel from huggingface model repository
TOKENIZER = AutoTokenizer.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
MODEL = AutoModel.from_pretrained("sberbank-ai/sbert_large_nlu_ru").to(DEVICE)


def l2_normalize(x):
  return x / np.linalg.norm(x, ord=2, axis=1, keepdims=True)


def get_bert_embeds(sentences, batch_size=1024):
  embeds = []
  for i in range(0, len(sentences), batch_size):
    sample = sentences[i:i + batch_size]

    encoded_input = TOKENIZER(sample, padding=True, truncation=True,
                              max_length=24, return_tensors='pt')
    encoded_input = to_device(encoded_input)

    with torch.no_grad():
      model_output = MODEL(**encoded_input)
    embeds.append(mean_pooling(model_output, encoded_input['attention_mask']).cpu().numpy())

  return l2_normalize(np.concatenate(embeds))


