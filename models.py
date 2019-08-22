import torch
import torch.nn as nn
from torch.distributions.dirichlet import Dirichlet

class Uncoiler(nn.Module):
  def __init__(self, net_):
    super(Uncoiler, self).__init__()
    mod_dict = self.get_module_dict(net_)
    for key in mod_dict:
      self.add_module(key, mod_dict[key])

  def get_module_dict(self, net_):
    mod_dict = {}
    for name, child in net_.named_children():
      lizt = [k for k in child.children()]
      if (len(lizt) == 0) or ('trans' in name):
        mod_dict[name] = child
        continue
      for name1, child1 in child.named_children():
        key = name+'_'+name1
        mod_dict[key] = child1
    return mod_dict

  def forward(self, x):
    child_length = sum(1 for _ in self.children())
    for i, mod in enumerate(self.children(), 1):
      if i == child_length:
        x = torch.nn.functional.avg_pool2d(x, x.size()[3])
        x = x.view(x.size(0), -1)
      x = mod(x)
    return(x)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()   
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x
  
class HydraNet(nn.Module):
  def __init__(self, model, n_heads=1, split_pt=7, num_classes=10, path=None):
    super(HydraNet, self).__init__()
    self.n_heads = n_heads
    self.split_pt = split_pt
    self.sample_wts = []
    for i in range(self.n_heads):
      self.sample_wts.append(Dirichlet(torch.tensor(np.ones(batch_size))).sample().float().to(device))
#       self.sample_wts.append(torch.tensor(np.ones(batch_size)*1).float().to(device))
    
    model_body = self.model_maker(model, num_classes)  
    if path != None:
      model_body.load_state_dict(torch.load(path)['state_dict'])
    model_body = Uncoiler(model_body)
    model_body = Uncoiler(model_body)
    self.layer_1 = nn.Sequential(*list(model_body.children())[:split_pt])
    self.layer_2 = nn.ModuleList()
    for i in range(self.n_heads):
      model_head = self.model_maker(model, num_classes) 
      model_head = Uncoiler(model_head)
      model_head = Uncoiler(model_head)
      makeBNWeighted(model_head, self.sample_wts[i])
      modules = list(model_head.children())[split_pt:-1]
      self.layer_2.append(nn.Sequential(*[*modules, Flatten(), list(model_head.children())[-1]]))
      
  def model_maker(self, model, num_classes):
    if model == "resnet":
      model_ = nn.DataParallel(eval(model)(depth=110, num_classes=num_classes)).cuda()
    elif model =="preresnet":
      model_ = nn.DataParallel(eval(model)(depth=110, num_classes=num_classes)).cuda()
    elif model =="densenet":
      model_ = nn.DataParallel(eval(model)(depth=100, num_classes=num_classes, growthRate=12)).cuda()
    else:
      raise ValueError("Does not recognise model name given")
    return model_
      
  def forward(self, x):
    x = self.layer_1(x)
    outputs = []
    for i in range(self.n_heads):
      outputs.append(self.layer_2[i](x))
    return outputs

  
