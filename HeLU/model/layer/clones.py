from torch import nn
import copy


### 重复模型模块

def clones(module, N):
    "Produce N identical layer."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

