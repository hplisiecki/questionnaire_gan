from generators import *
import numpy as np
import pandas as pd
from tqdm import tqdm

dataset = control_group()
n= 10


for i in tqdm(range(n-1)):
    dataset = pd.concat([dataset, control_group()])

for i in tqdm(range(n)):
    dataset = pd.concat([dataset, random_junk_group()])
    
for i in tqdm(range(n)):
    dataset = pd.concat([dataset, flat_junk_group()])
    
for i in tqdm(range(n)):
    dataset = pd.concat([dataset, ufo_junk_group()])
    
dataset.to_csv("data/dataset.csv")