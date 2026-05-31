import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"   # 强制走国内镜像
import torch
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

# 1. 准备数据加载逻辑
def load_data():
    categories = ['alt.atheism', 'soc.religion.christian'] #文本清洗：删除邮件头、签名、结尾、引用别人的话，得到纯正文文本
    train_data = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'))
    test_data = fetch_20newsgroups(subset='test', categories=categories, remove=('headers', 'footers', 'quotes'))
    
    return train_data.data, train_data.target, test_data.data, test_data.target

# 2. 定义 Dataset 类（为后续的DataLoader做前置准备）
class NewsgroupDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# 3. 计算评估指标的函数
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def main():
    # 加载原始文本
    train_texts, train_labels, test_texts, test_labels = load_data()

    # 初始化 Tokenizer (使用 bert-base-uncased)
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # 对文本进行 Tokenize
    # BERT 这里的处理包括了：分词、添加[CLS]/[SEP]、映射ID、Padding、截断
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128)

    # 封装为 Dataset 对象
    train_dataset = NewsgroupDataset(train_encodings, train_labels)
    test_dataset = NewsgroupDataset(test_encodings, test_labels)

    # 4. 加载预训练的 BERT 模型 (针对 2 分类任务)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # 5. 定义训练参数
    training_args = TrainingArguments(
        output_dir='./results',          # 输出目录
        num_train_epochs=3,              # 训练轮数
        per_device_train_batch_size=16,  # 训练批次大小
        per_device_eval_batch_size=64,   # 评估批次大小
        warmup_steps=100,                # 预热步数
        weight_decay=0.01,               # 权重衰减
        logging_dir='./logs',            # 日志目录
        logging_steps=10,
        eval_strategy="epoch",     # 每轮结束后评估
        save_strategy="epoch",
        load_best_model_at_end=True,     # 训练结束加载最优模型
    )

    # 6. 初始化 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    # 7. 开始训练
    print("开始微调模型...")
    trainer.train()

    # 8. 最终评估
    print("在测试集上评估...")
    eval_results = trainer.evaluate()
    print(f"评估结果: {eval_results}")

if __name__ == "__main__":
    main()