from flask import Flask,render_template,request
import torch
import numpy as np
import pytorch_lightning as pl
from transformers import BertModel
from torch import nn
from transformers import BertTokenizer
import os
app = Flask(__name__)

def sum(name):
    a=int(name[0])
    b=int(name[1])
    c=int(name[2])
    ans=a+b+c
    return ans

class BertForSequenceClassifier_pl(pl.LightningModule):
    def __init__(self, model_name, lr, num_class):
        super().__init__()
        self.save_hyperparameters()
        self.bert = BertModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_class)
        self.criterion = nn.CrossEntropyLoss()
        for param in self.bert.parameters():
            param.requires_grad = False
        for param in self.bert.encoder.layer[-1].parameters():
            param.requires_grad = True
def estimater(name):
    arg2mesh=[523871, 523872, 523873, 523874, 523875, 523876, 523877, 523970,
    523971, 523972, 523973, 523974, 523975, 523976, 523977, 533800,
    533802, 533803, 533804, 533805, 533806, 533807, 533811, 533812,
    533813, 533814, 533815, 533816, 533817, 533820, 533821, 533822,
    533823, 533824, 533825, 533826, 533827, 533830, 533831, 533832,
    533833, 533834, 533835, 533836, 533837, 533840, 533841, 533842,
    533843, 533844, 533845, 533846, 533847, 533850, 533851, 533852,
    533853, 533854, 533855, 533856, 533857, 533860, 533861, 533862,
    533863, 533864, 533865, 533866, 533867, 533870, 533871, 533872,
    533873, 533874, 533875, 533876, 533877, 533900, 533901, 533902,
    533903, 533904, 533905, 533906, 533907, 533910, 533911, 533912,
    533913, 533914, 533915, 533916, 533917, 533920, 533921, 533922,
    533923, 533924, 533925, 533926, 533927, 533930, 533931, 533932,
    533933, 533934, 533935, 533936, 533937, 533940, 533941, 533942,
    533943, 533944, 533945, 533946, 533947, 533950, 533951, 533952,
    533953, 533954, 533955, 533956, 533957, 533960, 533961, 533962,
    533963, 533964, 533965, 533966, 533967, 533970, 533971, 533972,
    533973, 533974, 533975, 533976, 533977, 543800, 543801, 543802,
    543803, 543804, 543805, 543806, 543807, 543810, 543811, 543812,
    543813, 543814, 543815, 543816, 543817, 543820, 543821, 543822,
    543823, 543824, 543825, 543826, 543827, 543837, 543900, 543901,
    543902, 543903, 543904, 543905, 543906, 543907, 543910, 543911,
    543912, 543913, 543914, 543915, 543916, 543917, 543920, 543921,
    543922, 543923, 543924, 543925, 543926, 543927, 544010, 544020]
    code_estimate_model_path = './models/model.ckpt'
    tokenizer = BertTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
    model = BertForSequenceClassifier_pl.load_from_checkpoint(code_estimate_model_path)
    bert=model.bert.cuda()
    classifier=model.classifier.cuda()
    text=name
    encoding = tokenizer(
    text,
    max_length = 107,           # 文章の長さを固定（Padding/Trancatinating）
    pad_to_max_length = True,# PADDINGで埋める
    truncation=True,
    padding = 'longest',
    return_tensors='pt')
    encoding = { k: v.cuda() for k, v in encoding.items() }
    with torch.no_grad():
        output = bert(**encoding)
        ans=classifier(output.pooler_output)
        ans = ans.to('cpu').detach().numpy().copy()
        ans=np.argmax(ans)
        ans=arg2mesh[ans]
    return ans

@app.route("/")
def index():
    name = request.args.get("name")
    return render_template("index.html",name=name)

@app.route("/index",methods=["post"])
def post():
    name = request.form["name"]
    name=estimater(name)
    return render_template("result.html", name=name)

if __name__ == "__main__":
    app.run(debug=True)