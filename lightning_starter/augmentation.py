import numpy as np
import torch

class CutMix(object):
  def __init__(self, size, beta):
    self.size = size
    self.beta = beta

  def __call__(self, batch):
    img, label = batch
    rand_img, rand_label = self._shuffle_minibatch(batch)
    lambda_ = np.random.beta(self.beta,self.beta)
    r_x = np.random.uniform(0, self.size)
    r_y = np.random.uniform(0, self.size)
    r_w = self.size * np.sqrt(1-lambda_)
    r_h = self.size * np.sqrt(1-lambda_)
    x1 = int(np.clip(r_x - r_w // 2, a_min=0, a_max=self.size))
    x2 = int(np.clip(r_x + r_w // 2, a_min=0, a_max=self.size))
    y1 = int(np.clip(r_y - r_h // 2, a_min=0, a_max=self.size))
    y2 = int(np.clip(r_y + r_h // 2, a_min=0, a_max=self.size))
    img[:, :, x1:x2, y1:y2] = rand_img[:, :, x1:x2, y1:y2]
    
    lambda_ = 1 - (x2-x1)*(y2-y1)/(self.size*self.size)
    return img, label, rand_label, lambda_

  def _shuffle_minibatch(self, batch):
    img, label = batch
    rand_img, rand_label = img.clone(), label.clone()
    rand_idx = torch.randperm(img.size(0))
    rand_img, rand_label = rand_img[rand_idx], rand_label[rand_idx]
    return rand_img, rand_label

# Code: https://github.com/facebookresearch/mixup-cifar10
class MixUp(object):
  def __init__(self, alpha=0.1):
    self.alpha = alpha

  def __call__(self, batch):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    x, y = batch
    lam = np.random.beta(self.alpha, self.alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam
