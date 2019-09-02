import numpy as np
import torch
import torch.nn as nn




# Function taken from here https://github.com/Bjarten/early-stopping-pytorch
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss


class EarlyStoppingExtended(EarlyStopping):
  def __init__(self, patience):
    super(EarlyStoppingExtended, self).__init__(patience=patience, verbose=False)
    self.val_acc = 0.0
    self.train_acc = 0.0
    self.train_loss = 0.0
    self.epochs = 0
  
  def __call__(self, val_loss, val_acc, train_loss, train_acc, epochs, model):

    score = -val_loss

    if self.best_score is None:
        self.best_score = score
        self.save_checkpoint(val_loss, val_acc, train_loss, train_acc, epochs, model)
    elif score < self.best_score:
        self.counter += 1
        print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
        if self.counter >= self.patience:
            self.early_stop = True
    else:
        self.best_score = score
        self.save_checkpoint(val_loss, val_acc, train_loss, train_acc, epochs, model)
        self.counter = 0

  def save_checkpoint(self, val_loss, val_acc, train_loss, train_acc, epochs, model):
    '''Saves model when validation loss decrease.'''
    if self.verbose:
        print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
    torch.save(model.state_dict(), 'checkpoint.pt')
    self.val_loss_min = val_loss
    self.val_acc = val_acc
    self.train_acc = train_acc
    self.train_loss = train_loss
    self.epochs = epochs



class BatchNormWeighted(nn.BatchNorm2d):
    def __init__(self, num_features, wts):
      super(BatchNormWeighted, self).__init__(num_features)
      self.wts = wts
      
    def forward(self, x):
      self._check_input_dim(x)
      y1 = ((x.view(x.size(0), -1)*self.wts[:(x.view(x.size(0), -1)).shape[0], None])/self.wts[:(x.view(x.size(0), -1)).shape[0]].sum()).reshape(x.shape)
      y1 = y1.transpose(0,1)
      y = x.transpose(0,1)
      return_shape = y.shape
      y1 = y1.contiguous().view(x.size(1), -1)
      y = y.contiguous().view(x.size(1), -1)
      mu = y1.mean(dim=1)
      sigma2 = y1.var(dim=1)
      if self.training is not True:
          y = y - self.running_mean.view(-1, 1)
          y = y / (self.running_var.view(-1, 1)**.5 + self.eps)
      else:
          if self.track_running_stats is True:
              with torch.no_grad():
                  self.running_mean = (1-self.momentum)*self.running_mean + self.momentum*mu
                  self.running_var = (1-self.momentum)*self.running_var + self.momentum*sigma2
          y = y - mu.view(-1,1)
          y = y / (sigma2.view(-1,1)**.5 + self.eps)

      y = self.weight.view(-1, 1) * y + self.bias.view(-1, 1)
      return y.view(return_shape).transpose(0,1)


def freeze_body(net_, state=True):
  ct = 0
  for name, child in net_.named_children():
    for name, child2 in child.named_children():
      if(name == 'layer_2'):
        break
      for param_name, params in child2.named_parameters():
        params.requires_grad = not state
        ct += 1


def model_breakup(net_):
  ct = 0
  for child in net_.children():
    print('#################################################################################', ct)
    print(child)
    ct += 1


def makeBNWeighted(net_, wts):
  lizt = [k for k in net_.children()]
  if len(lizt) == 0:
    return
  for name, child in net_.named_children():
    if isinstance(child, nn.BatchNorm2d):
      setattr(net_, name, BatchNormWeighted(child.num_features, wts=wts))
    makeBNWeighted(child, wts)


def uncoil(net_, ct=-2):
  lizt = [k for k in net_.children()]
  ct += 1
  if len(lizt) == 0:
    print(' ###'*ct, net_)
    pass
    return
  for child in net_.children():
    uncoil(child, ct)


def uncoil_names(net_, ct=-1):
  lizt = [k for k in net_.children()]
  ct += 1
  if len(lizt) == 0:
    return
  for name, child in net_.named_children():
    print(' ##'*ct, name)
    uncoil_names(child, ct)