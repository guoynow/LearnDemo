from datasets import load_dataset
from accelerate import Accelerator, DeepSpeedPlugin

from transformers import (
    AutoTokenizer, 
    AutoModel, 
    AutoModelForSequenceClassification,
    AutoConfig,
    get_cosine_schedule_with_warmup,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from peft import get_peft_model, PrefixTuningConfig, LoraConfig, TaskType

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm.auto import tqdm

# 固定seed
torch.manual_seed(42)
# 确定设备
accelerator = Accelerator()
device = accelerator.device

num_epochs = 5
patience = 5

training_record = {}

# tokenizer，加载bert的分词器,uncased就是不区分大小写
tokenizer = AutoTokenizer.from_pretrained("/home/guoynow/code/DL/chapter_9-15/bert-base-uncased")

# 加载数据里
dataset_sst2 = load_dataset(
    "parquet",
    data_files={
        "train": "./sst2/data/train-00000-of-00001.parquet",
        "validation": "./sst2/data/validation-00000-of-00001.parquet"
        })

# preprocessing
def collate_fn(batch):
    #对字符串文本，进行编码，变为id,longest就是最长，padding就是填充,truncation为True就是截断
    inputs = tokenizer([x["sentence"] for x in batch], padding="longest", truncation=True, return_tensors="pt", max_length=512)
    labels = torch.tensor([x["label"] for x in batch])
    return inputs, labels

train_loader = DataLoader(dataset_sst2["train"], batch_size=128, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(dataset_sst2["validation"], batch_size=128, collate_fn=collate_fn)

def evaluate(model, val_loader):
    model.eval()
    val_loss = 0
    val_acc = 0
    with torch.no_grad():# 在评估过程中关闭梯度计算
        total_samples = 0 #统计验证集总样本数量
        for inputs, labels in val_loader:
            inputs = {k: v.to(device) for k, v in inputs.items()} #输入是一个字典，所以拿value
            labels = labels.to(device)
            probs = model(**inputs)
            probs = probs.logits.squeeze()
            probs, labels = accelerator.gather_for_metrics((probs, labels))
            loss = F.cross_entropy(probs, labels.float()) #求损失
            val_loss += loss.item()
            val_acc += ((probs > 0.5) == labels).sum().item() #模型的预测结果与实际标签是否相等,求和得到预测正确数量
            total_samples += len(labels)

    val_loss /= len(val_loader)
    val_acc /= total_samples
    return val_loss, val_acc


def train(model, train_loader, val_loader, device, num_epochs=3, patience=3):
    # 将模型移动到指定设备
    model.to(device)

    # 定义优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    # 计算训练步数总数
    total_steps = num_epochs * len(train_loader)

    # 使用transformers库中的余弦学习率调度器进行学习率调整
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.2 * total_steps), #前20%步，学习率提升
        num_training_steps=total_steps
    )

    model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)
    # 提前停止训练的控制变量
    best_val_acc = -1
    cur = 0

    # 存储训练和验证指标的容器
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in tqdm(range(num_epochs)):
        # 进入训练模式
        model.train()
        train_loss = 0
        train_acc = 0
        total_samples = 0

        # 对训练数据进行迭代
        for inputs, labels in train_loader:
            # 将数据移动到指定设备
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)
            
            # 前向传播并计算损失
            optimizer.zero_grad()
            probs = model(**inputs) # **代表字典解包，inputs 中的键名必须与模型 forward() 方法的参数名完全一致
            probs = probs.logits.squeeze()
            loss = F.binary_cross_entropy_with_logits(probs, labels.float())
            # loss.backward()
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()

            # 收集指标
            train_loss += loss.item()
            train_acc += ((probs > 0.5) == labels).sum().item()
            total_samples += len(labels)

        train_loss /= len(train_loader)
        train_acc  /= total_samples

        # 进行验证
        val_loss, val_acc = evaluate(model, val_loader)

        # 记录指标
        print(f"epoch {epoch}: train_loss {train_loss:.4f}, train_acc {train_acc:.4f}, val_loss {val_loss:.4f}, val_acc {val_acc:.4f}")
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # 提前停止训练
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            cur = 0
        else:
            cur += 1
        if cur >= patience:
            print("提前停止训练")
            break

    return history

model = AutoModelForSequenceClassification.from_pretrained("/home/guoynow/code/DL/chapter_9-15/bert-base-uncased", num_labels=1)
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,    # 序列分类
    r=16,
    lora_alpha=16,
    lora_dropout=0.1,
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()  # 查看可训练参数（<1%）

training_record["LoRA"] = train(model, train_loader, val_loader, device, num_epochs=10, patience=5)