from fullModel import RobertaClass
import torch 

model = torch.load('3labelmodel')
print(model)