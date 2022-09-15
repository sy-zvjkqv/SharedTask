import os
import numpy as np
import pandas as pd
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import logging
from sklearn.model_selection import train_test_split
import torch
import random
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
import pytorch_lightning as pl
from sklearn.preprocessing import LabelEncoder
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel
from transformers import BertModel
text=input("入力してください:")
tokenizer = BertTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
class BertForSequenceClassifier_pl(pl.LightningModule):
    def __init__(self, model_name, lr, num_class):
        super().__init__()
        self.save_hyperparameters()
        self.bert = BertModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_class)
        self.criterion = nn.CrossEntropyLoss()

def estimation(text):
    label2mesh=[523871, 523872, 523873, 523874, 523875, 523876, 523877, 523970,523971, 523972, 523973, 523974, 523975, 523976, 523977, 533800,533802, 533803, 533804, 533805, 533806, 533807, 533811, 533812,533813, 533814, 533815, 533816, 533817, 533820, 533821, 533822,533823, 533824, 533825, 533826, 533827, 533830, 533831, 533832,33833, 533834, 533835, 533836, 533837, 533840, 533841, 533842,533843, 533844, 533845, 533846, 533847, 533850, 533851, 533852,533853, 533854, 533855, 533856, 533857, 533860, 533861, 533862,533863, 533864, 533865, 533866, 533867, 533870, 533871, 533872,533873, 533874, 533875, 533876, 533877, 533900, 533901, 533902,533903, 533904, 533905, 533906, 533907, 533910, 533911, 533912,533913, 533914, 533915, 533916, 533917, 533920, 533921, 533922,533923, 533924, 533925, 533926, 533927, 533930, 533931, 533932,533933, 533934, 533935, 533936, 533937, 533940, 533941, 533942,533943, 533944, 533945, 533946, 533947, 533950, 533951, 533952,533953, 533954, 533955, 533956, 533957, 533960, 533961, 533962,533963, 533964, 533965, 533966, 533967, 533970, 533971, 533972,533973, 533974, 533975, 533976, 533977, 543800, 543801, 543802,543803, 543804, 543805, 543806, 543807, 543810, 543811, 543812,543813, 543814, 543815, 543816, 543817, 543820, 543821, 543822,543823, 543824, 543825, 543826, 543827, 543837, 543900, 543901,543902, 543903, 543904, 543905, 543906, 543907, 543910, 543911,543912, 543913, 543914, 543915, 543916, 543917, 543920, 543921,543922, 543923, 543924, 543925, 543926, 543927, 544010, 544020]
    model = BertForSequenceClassifier_pl.load_from_checkpoint("/home/is/shuntaro-o/miniconda3/SharedTask_main/Flask/epoch=2-step=2400000.ckpt")
    bert=model.bert.cuda()
    classifier=model.classifier.cuda()
    encoding = tokenizer(
    text,
    padding = 'longest',
    return_tensors='pt')
    encoding = { k: v.cuda() for k, v in encoding.items() }
    with torch.no_grad():
        output = bert(**encoding)
        ans=classifier(output.pooler_output)
        ans = ans.to('cpu').detach().numpy().copy()
        ans=np.argmax(ans)
        ans=label2mesh[ans]
    return ans

ans=estimation(text)
print(ans)
