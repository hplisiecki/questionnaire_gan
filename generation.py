from generators import *
import numpy as np
import pandas as pd

dataset = control_group()
n= 10


for i in range(n-1):
    dataset = pd.concat([dataset, control_group()])

for i in range(n):
    dataset = pd.concat([dataset, random_junk_group()])
    
for i in range(n):
    dataset = pd.concat([dataset, flat_junk_group()])
    
for i in range(n):
    dataset = pd.concat([dataset, ufo_junk_group()])
    
print(dataset.shape)