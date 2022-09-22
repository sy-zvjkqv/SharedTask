import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from transformers import BertTokenizer

MODEL_NAME = "cl-tohoku/bert-base-japanese-whole-word-masking"
batch = 1
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)


def Data_pre(df):
    index_train = int(len(df) * 0.8)
    index_val = int(len(df) * 0.1)
    index_test = int(len(df) * 0.1)
    df["tweet"] = df["tweet"].astype(str)
    df_train = df.iloc[0:index_train, 1:]
    df_val = df.iloc[index_train : index_train + index_val, 1:]
    df_test = df.iloc[
        index_train + index_val : index_train + index_val + index_test, 1:
    ]
    L = LabelEncoder()
    df["code"] = L.fit_transform(df["code"])
    df_train["code"] = L.transform(df_train["code"])
    df_test["code"] = L.transform(df_test["code"])
    df_val["code"] = L.transform(df_val["code"])
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    num_class = df_train["code"].max() + 1
    return df_train, df_val, df_test, num_class


def Dataloader(df, x, y, batch):
    sentences_text = df[x].to_list()
    label_list = df[y].to_list()
    dataset_for_loader = []
    i = 0
    for sentence in sentences_text:
        encoding = tokenizer(
            sentence,
            max_length=100,  # 文章の長さを固定（Padding/Trancatinating）
            pad_to_max_length=True,  # PADDINGで埋める
            truncation=True,
            # return_tensors = 'pt'
        )
        encoding["labels"] = label_list[i]
        i = i + 1
        encoding = {k: torch.tensor(v) for k, v in encoding.items()}
        dataset_for_loader.append(encoding)
    dataloader = DataLoader(dataset_for_loader, batch_size=batch)
    return dataloader
