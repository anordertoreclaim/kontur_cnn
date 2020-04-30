import youtokentome as yttm
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
import os

from functools import partial
from scipy.optimize import minimize

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch

PAD_ID = 0

def precision(preds, trues):
  preds = np.array(preds)
  trues = np.array(trues)
  return np.sum(preds[trues == 1]) / np.sum(preds)

def recall(preds, trues):
  preds = np.array(preds)
  trues = np.array(trues)
  return np.sum(preds[trues == 1]) / np.sum(trues)

def f_score(preds, trues):
  p = precision(preds, trues)
  r = recall(preds, trues)
  return 2 * p * r / (p + r)
  
def split_train(train_df, group_by, train_size=0.75, thresh_size=0.3, seed=42):
    assert(group_by in ['ru_name', 'eng_name'])
    np.random.seed(seed)
    gss = GroupShuffleSplit(1, train_size=train_size, random_state=seed)
    groups = getattr(train_df, group_by)
    train_idx, val_idx = next(gss.split(train_df, groups=groups))
    val_df = train_df.iloc[val_idx]
    train_df = train_df.iloc[train_idx] 

    val_df = val_df.reset_index(drop=True)
    gss = GroupShuffleSplit(1, train_size=thresh_size, random_state=seed)
    groups = getattr(val_df, group_by)
    thresh_idx, holdout_idx = next(gss.split(val_df, groups=groups))
    thresh_df, holdout_df = val_df.iloc[thresh_idx], val_df.iloc[holdout_idx]
    
    return train_df, thresh_df, holdout_df
    
def transform_dataframe(df, transform, save_as=None):
  df = df.copy()
  df.ru_name = df.ru_name.apply(transform)
  df.eng_name = df.eng_name.apply(transform)
  if save_as is not None:
    df.to_csv(save_as)
  return df

def train_bpe_tokenizer(df, tokenizer_filename, vocab_size):
  with open('train.txt', 'w') as fout:
    for col in ['ru_name', 'eng_name']:
        for name in df[col]:
            print(name, file=fout)
  yttm.BPE.train(data='train.txt', vocab_size=vocab_size, model=tokenizer_filename)
  os.remove('train.txt')
    
def precompute_dataset(df, tokenizer):
  ru_tokens = df.ru_name.apply(lambda ru: tokenizer.encode([ru], 
                                                           output_type=yttm.OutputType.ID, 
                                                           bos=False,
                                                           eos=False)[0])
  eng_tokens = df.eng_name.apply(lambda eng: tokenizer.encode([eng], 
                                                              output_type=yttm.OutputType.ID, 
                                                              bos=False,
                                                              eos=False)[0])
  
  ru_tokens = list(ru_tokens)
  eng_tokens = list(eng_tokens)

  if 'answer' in df.columns:
    answers = df.answer.apply(lambda ans: 1 if ans == True else 0)
    return ru_tokens, eng_tokens, list(answers)
  
  return ru_tokens, eng_tokens


class ListsDataset(Dataset):
    def __init__(self, *lists):
      assert all(len(lists[0]) == len(l) for l in lists)
      self.lists = lists
        
    def __getitem__(self, idx):
      return tuple([l[idx] for l in self.lists])
        
    def __len__(self):
      return len(self.lists[0])


def collate_fn(batch, with_labels=True):
    if with_labels:
        (ru, eng, labels) = zip(*batch)
    else:
        (ru, eng) = zip(*batch)

    ru = [torch.tensor(el) for el in ru]
    eng = [torch.tensor(el) for el in eng]
    ru_pad = pad_sequence(ru, batch_first=True, padding_value=PAD_ID)
    eng_pad = pad_sequence(eng, batch_first=True, padding_value=PAD_ID)
    if with_labels:
        labels = torch.tensor(labels, dtype=int)
        return ru_pad, eng_pad, labels
    else:
        return ru_pad, eng_pad


class OptimizedRounder:
    """
    An optimizer for rounding thresholds
    to maximize F1 score
    # https://www.kaggle.com/naveenasaithambi/optimizedrounder-improved
    """
    def __init__(self, initial_coef=0.3):
        self.thresh_ = None
        self.initial_coef = initial_coef

    def _f_score_loss(self, threshold, similarities, trues):
        """
        Get loss according to
        using current coefficients
        
        :param threshold: Prediction threshold
        :param similarities: Predicted cosine similarities
        :param true: The ground truth labels
        """
        preds = pd.cut(similarities, [-np.inf] + list(threshold) + [np.inf], labels = [0, 1])

        return -f_score(preds, trues)

    def fit(self, similarities, trues):
        """
        Optimize prediction threshold
        
        :param X: The raw predictions
        :param y: The ground truth labels
        """
        loss_partial = partial(self._f_score_loss, similarities=similarities, trues=trues)
        self.thresh_ = minimize(loss_partial, self.initial_coef, method='nelder-mead')['x']

    def predict(self, similarities):
        """
        Make predictions with optimized threshold
        
        :param threshold: Prediction threshold
        :param similarities: Predicted cosine similarities
        """
        return pd.cut(similarities, [-np.inf] + list(self.thresh_) + [np.inf], labels = [0, 1])

