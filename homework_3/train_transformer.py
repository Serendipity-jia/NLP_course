import os
import math
import random
import argparse
from pathlib import Path
from typing import List, Tuple

import sentencepiece as spm
import sacrebleu
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from Transformer import *

#============
#全局性能&线程设置
#=============
import multiprocessing
#设置Pytorch使用的CPU线程数，由于我租用的GPU所使用的CPU核心数是16，所以这里设置成了16
torch.set_num_threads(16)           
# cudnn 优化：让GPU自动寻找最快的卷积算法，大幅加速训练
torch.backends.cudnn.benchmark = True

#=================
#配置/超参数
#=================
DEFAULTS = {
    #数据文件路径
    "train_file": "data/eng-fra_train_data.txt",
    "test_file": "data/eng-fra_test_data.txt",
    #模型基础结构超参数
    #Train pairs: 108673 Test pairs 27169;根据数据量选择适当的超参数和正则化手段避免过拟合
    "sp_model": "spm.model",#训练好的子词分词器
    "vocab_size": 8000,
    "d_model": 256,
    "d_ff": 1024,
    "heads": 4,
    "num_layers": 4,
    "dropout": 0.2,
    #训练参数
    "batch_size": 64,  #句子数batch，24GB 可尝试 64-256，根据 avg length 调整
    #最好按token打包：句子长短不一，按句子打包会产生大量无效padding，浪费GPU；
    #按token打包能自动把长度相近的句子打包在一起，从而最小化padding。
    "num_workers":12, # 16 核 CPU，可试 10-14，避免占满系统全部核
    "pin_memory": True,
    "persistent_works": True,
    "max_len": 100, #句子的最大长度：超过100个token会被截断
    "epochs": 60,   #训练轮数：把全部训练数据过60遍，配合早停
    "warmup": 1000, #学习率预热步数：前1000步慢慢提升学习率
    "label_smoothing": 0.1, #标签平滑，正则化手段，防止模型过于自信
    "early_stop_patience": 5, #允许连续多少个 epoch 不提升后停止
    "early_stop_min_delta": 0.0, #认为有“提升”的最小 BLEU 增量
    #运行环境
    "device": "cuda",
    "save_dir": "checkpoints",
    "seed": 42,
    #翻译参数
    "beam_size": 5,#beem search的宽度
    "max_decoding_len": 100,#翻译输出的最大长度
}

# reproducibility
# 初始化模型参数是随机的；dropout是随机的；数据集shuffle是随机的
def set_seed(seed):
    random.seed(seed) #固定Python自带随机库的随机数：数据集shuffle
    torch.manual_seed(seed) #固定Pytorch CPU 上的随机数：模型参数初始化、dropout
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        #固定所有GPU上的随机数：如果有可用的GPU，把模型放到GPU上，那么模型参数初始化、dropout就会在GPU上执行

#===============
#数据预处理
#===============

# 读取机器翻译的英法成对预料，返回句子对列表
def read_parallel(file_path: str, max_lines: int=None) -> List[Tuple[str, str]]:
    pairs=[]
    #读文本文件必须指定utf-8：utf-8 是全球唯一通用编码，支持英法中所有语言
    with open(file_path, "r", encoding="utf=8") as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            #去掉首位空格、换行
            line = line.strip()
            parts = line.split("\t")
            if len(parts) < 2:
                continue

            src, tgt = parts[0].strip(), parts[1].strip()
            pairs.append((src, tgt))
    return pairs

#合并训练集、测试集语料为一个临时文件用于训练sentencepiece
def train_sentencepiece(corpus_files: List[str], model_prefix: str, vocab_size: int):
    tmp = model_prefix + ".corpus.tmp"
    #将训练集语料写入到临时文件
    with open(tmp, "w", encoding="utf-8") as out:
        for f in corpus_files:
            with open(f, "r", encoding="utf-8") as fr:
                for line in fr:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split("\t")
                    if len(parts) < 2:
                        continue
                    #英法双语语料放到同一个语料库，后面也共用同一个词表
                    out.write(parts[0].strip() + "\n")
                    out.write(parts[1].strip() + "\n")
    #用之前写好的语料库训练SentencePiece子词分词模型，生成统一的英法共用词表
    spm.SentencePieceTrainer.Train(
        input=tmp,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=1.0, #覆盖100%的字符，这个设置适合欧洲语言（英法德西）
        model_type="unigram", #主流
        bos_id=1, eos_id=2, unk_id=0, pad_id=3 #固定这四个token的id
    )
    os.remove(tmp)

#==============
# Dataset & Collate
#==============
#把之前处理好的句子对转化为数字张量
class TranslationDataset(Dataset):
    #继承Dataset，就可以使用DataLoader加载批量数据
    def __init__(self, pairs: List[Tuple[str, str]], sp: spm.SentencePieceProcessor, max_len: int):
        self.pairs = pairs
        self.sp = sp
        self.max_len = max_len
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        #列表
        src_ids = self.sp.encode(src, out_type=int) 
        tgt_ids = self.sp.encode(tgt, out_type=int)
        # Add bos/eos: sentencepiece model used bos_id=1 eos_id=2 per training above
        src_ids = [1] + src_ids[: self.max_len - 2] + [2]
        tgt_ids = [1] + tgt_ids[: self.max_len - 2] + [2]
        #返回pytorch张量
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)

def collate_fn(batch, pad_id=3, tokens_per_batch=4096):
    "使用DataLoader加载数据时对一个batch的处理函数：把一批长短不一的句子，统一用padding补齐长度，并生成注意力掩码，打包成Transformer"
    "能直接训练的标准格式"
    #batch:list of (src_tensor, tgt_tensor)
    srcs = [x[0] for x in batch]
    tgts = [x[1] for x in batch]
    src_lens = [len(s) for s in srcs]
    tgt_lens = [len(t) for t in tgts]
    max_src = max(src_lens)
    max_tgt = max(tgt_lens)
    #创建全是<pad>的空张量
    src_padded = torch.full((len(batch), max_src), pad_id, dtype=torch.long)
    tgt_padded = torch.full((len(batch), max_tgt), pad_id, dtype=torch.long)
    #把真实句子填进去，剩下保持<pad>
    for i, s in enumerate(srcs):
        src_padded[i, : len(s)] = s
    for i, t in enumerate(tgts):
        tgt_padded[i, : len(t)] = t
    #mask(batch, 1 , max_len)
    #这里mask的构造来自与batch相关的信息，而batch会被输送到GPU上，所以mask也在GPU上
    src_mask = (src_padded != pad_id).unsqueeze(-2)  
    tgt_mask = (tgt_padded != pad_id).unsqueeze(-2)
    #未来信息掩码，没有指定device，新建的张量永远默认在CPU！
    size = max_tgt
    subsequent = torch.triu(torch.ones((1, size, size), dtype=torch.uint8), diagonal=1)
    subsequent = (subsequent == 0)
    #将subsequence移动到tgt_mask所在设备，并合并两个掩码
    tgt_mask = tgt_mask & subsequent.to(tgt_mask.device)
    return src_padded, tgt_padded, src_mask, tgt_mask
# 顶层位置：定义一次: lambda/局部函数不可 picklable，DataLoader worker 需要可序列化回调。
# 把 collate 函数转为模块顶层函数或使用 functools.partial（且被包裹的函数是顶层的），或在调试时将 num_workers 设为 0，均可解决问题。
def collate_with_pad3(batch):
    return collate_fn(batch, pad_id=3)

#===========
# Loss & Scheduler
#===========
class LabelSmoothingloss(nn.Module):
    def __init__(self, label_smoothing, tgt_vocab, ignore_index=3):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction="sum") #计算带标签平滑的KL散度损失
        self.padding_idx = ignore_index
        self.confidence = 1.0 - label_smoothing
        self.smoothing = label_smoothing
        self.tgt_vocab = tgt_vocab
    def forward(self, x, target):
        #x:预测概率分布：(batch, max_len, vocab_size)
        #targrt:（batch, max_len)
        x = x.view(-1, self.tgt_vocab) #(batch*max_len, vocab_size)，方便计算损失
        target = target.reshape(-1) #(batch*max_len),方便计算损失
        #创建空分布，用来存放平滑后的真实标签
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.tgt_vocab - 2))
        mask = (target == self.padding_idx)
        #传入两个整数数组索引选取正确词，并给正确词设置高概率
        true_dist[torch.arange(len(target)), target] = self.confidence
        #把padding位置的概率分布全部设为全0，让损失完全不计算padding位置的预测损失
        true_dist[mask] = 0 
        return self.criterion(x, true_dist) / (~mask).sum().clamp(min=1)
        #平均每个有效词的损失， clamp(min=1)防止除0
    
class NoamOpt:
    "Noam 学习率调度,在Warmup阶段线性递增学习率，后续阶段则按公式递减学习率"
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer #传入优化器Adam
        self._step = 0 #记录当前训练步数
        self.warmup = warmup 
        self.factor = factor #缩放系数
        self.model_size = model_size
        self._rate = 0 #记录当前学习率
    def step(self):
        self._step += 1
        rate = self.rate()
        #param_groups 是参数组列表，默认只有1组（装全部参数），
        #循环是代码的鲁棒写法，确保当对参数进行手动分组的时候每个组的学习率都能被更新到
        for p in self.optimizer.param_groups:
            p["lr"] = rate #把新的学习率设置给优化器
        self._rate = rate 
    def zero_grad(self):
        self.optimizer.zero_grad()
    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** -0.5) * min(step ** -0.5, step * self.warmup ** -1.5)

#==============
#解码 & 评估
#即便是在 inference（greedy/beam）阶段，通常仍把 batch 中的 src/tgt 按最大长度 pad，因此 mask 仍然必要以阻止模型“看到”pad。
#但下面实现的greedy_decode和beam_search_decode是基于batch=1，可以不用考虑这个问题，但也加上了padding掩码逻辑
#==============         #传入训练好的模型，和待预测的英文序列的token IDs
def greedy_decode(model, sp, src_tok, src_mask, max_len, device):
    model.eval()#关闭dropout
    memory = model.encode(src_tok.to(device), src_mask.to(device))#(1, max_len)
    #初始化输出序列(从<bos>开始，索引为1)
    ys = torch.tensor([[1]], dtype=torch.long, device=device) #(1,)
    #循环逐次生成法文
    for i in range(max_len - 1):
        tgt_mask = (ys != 3).unsqueeze(-2) #(1, 1, size)
        size = ys.size(1)
        subsequent = torch.triu(torch.ones((1, size, size), dtype=torch.uint8), diagonal=1)
        subsequent = (subsequent == 0).to(device) 
        tgt_mask = tgt_mask & subsequent #广播->(1, size, size)
        out = model.decode(memory, src_mask.to(device), ys, tgt_mask)#(1, len(ys), d_model)
        prob = model.generator(out[:, -1]) #只取最后一个词的预测概率分布(1, d_model)
        _, next_word = torch.max(prob, dim=1)#next:(1,)
        ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1) #(1, size+1)
        if (next_word == 2).all():
            break
    return ys 

def batched_greedy_decode(model, sp, src, src_mask, max_len, device, use_autocast=False):
    model.eval()
    src = src.to(device)
    src_mask = src_mask.to(device)
    with torch.no_grad():
        if use_autocast and device.startswith("cuda"):
            with torch.amp.autocast("cuda"):
                memory = model.encode(src, src_mask) #(batch, seq_len, d_model)
        else:
            memory = model.encode(src, src_mask)
    batch = src.size(0)
    #初始化输出序列
    ys = torch.full((batch, 1), 1, dtype=torch.long, device=device) #(batch, 1)
    finished = torch.zeros(batch, dtype=torch.bool, device=device) #（batch,)
    for _ in range(max_len - 1):
        size = ys.size(1) #目前已生成序列的长度
        # subsequent: (1, size, size) bool
        subsequent = torch.triu(torch.ones((1, size, size), dtype=torch.uint8, device=device), diagonal=1)
        subsequent = (subsequent == 0)  # bool
        # tgt_mask: (B, size) -> (B,1,size) & (1,size,size) -> broadcast (B, size, size)
        tgt_mask = (ys != 3).unsqueeze(-2) & subsequent  

        with torch.no_grad():
            if use_autocast and device.startswith("cuda"):
                with torch.amp.autocast(device_type="cuda"):
                    out = model.decode(memory, src_mask, ys, tgt_mask)  # (B, size, d_model)
            else:
                out = model.decode(memory, src_mask, ys, tgt_mask)
            
            logits = model.generator(out[:, -1])  # (B, d_model)
            # 选择概率最大的 token
            next_tokens = logits.argmax(dim=-1)  # (B,)

            # 对已完成序列，不再改变（填 pad）
            next_tokens = torch.where(finished, torch.full_like(next_tokens, 3), next_tokens)
        ys = torch.cat([ys, next_tokens.unsqueeze(1)], dim=1) #(B, size+1)
        #标记该batch中哪些句子已经遇到<eos>结束了
        finished = finished | (next_tokens == 2)
        if finished.all():
            break
    return ys #包含bos （batch, max_len)，用<pad>补足长度来对齐

def beam_search_decode(model, sp, src_tok, src_mask, max_len, beam_size, device):
    # 简单实现：逐步扩展 top-K（非最优但可用）
    model.eval()
    memory = model.encode(src_tok.to(device), src_mask.to(device))
    batch = src_tok.size(0)
    # For simplicity implement beam for batch=1
    assert batch == 1, "beam_search currently supports batch=1"
    hypotheses = [(torch.tensor([1], dtype=torch.long, device=device), 0.0)]  # (seq, score)
    for _ in range(max_len-1):
        new_hyps = []
        for seq, score in hypotheses:
            ys = seq.unsqueeze(0)
            tgt_mask = (ys != 3).unsqueeze(-2)
            size = ys.size(1)
            subsequent = torch.triu(torch.ones((1, size, size), dtype=torch.uint8), diagonal=1)
            subsequent = (subsequent == 0).to(device)
            tgt_mask = tgt_mask & subsequent
            out = model.decode(memory, src_mask.to(device), ys, tgt_mask)
            logp = model.generator(out[:, -1]).squeeze(0)  # (V,)
            topk = torch.topk(logp, beam_size)
            for k in range(beam_size):
                w = topk.indices[k].item()
                sc = score + topk.values[k].item()
                new_seq = torch.cat([seq, torch.tensor([w], dtype=torch.long, device=device)])
                new_hyps.append((new_seq, sc))
        # keep top beam_size
        new_hyps = sorted(new_hyps, key=lambda x: x[1], reverse=True)[:beam_size]
        hypotheses = new_hyps
        # early stop if all top hypotheses ended
        if all(h[0][-1].item() == 2 for h in hypotheses):
            break
    best = hypotheses[0][0].cpu().numpy().tolist()
    return torch.tensor(best, dtype=torch.long).unsqueeze(0)

def ids_to_str(sp, ids):
    #remove bos/eos/pad tokens(1, 2, 3)
    #还需要删除<pad>是因为在解码时为了batch长度对齐会用<pad>补齐，所以这里需要处理掉
    #虽然后续在预测的时候设定的batch=1，可以不用考虑这个问题，但仍然保留了这个处理逻辑
    seq = [i for i in ids if i not in (1, 2, 3)]
    return sp.decode(seq)

#=======================
# 训练主函数
#=======================
def train_main(args):
    set_seed(args["seed"])
    os.makedirs(args["save_dir"], exist_ok=True)

    #1.读数据
    print("Reading data...")
    train_pairs = read_parallel(args["train_file"])
    test_pairs = read_parallel(args["test_file"])
    train_pairs = train_pairs[:130]
    test_pairs = test_pairs[:40]
    print("Train pairs:", len(train_pairs), "Test pairs", len(test_pairs))
    #2.训练或加载sentencepiece
    if not os.path.exists(args["sp_model"]):
        print("Training SentencePiece model...")
        train_sentencepiece([args["train_file"]],
                            args["sp_model"].replace(".model", ""),
                            args["vocab_size"])
    sp = spm.SentencePieceProcessor()
    sp.Load(args["sp_model"])

    vocab_size = sp.get_piece_size()
    print("Loaded sp model. Vocab size:", vocab_size)

    #3.Dataset/Dataloader(简单按句子batch)
    train_dataset = TranslationDataset(train_pairs, sp, args["max_len"])
    test_dataset = TranslationDataset(test_pairs, sp, args["max_len"])
    #按句子patch
    train_loader = DataLoader(train_dataset, 
                            batch_size=args["batch_size"],
                            shuffle=True, 
                            collate_fn=collate_with_pad3, 
                            num_workers=args["num_workers"], #开启多个子进程并行加载数据，不让GPU空闲等待。
                            pin_memory=args.get("pin_memory", True),#给GPU开高速通道，CPU→GPU传输更快，GPU训练必开
                            persistent_workers=args.get("persistent_workers", True),) #数据加载进程不关闭、重复使用，避免反复创建销毁进程，大幅提速。
    val_loader = DataLoader(test_dataset, 
                            batch_size=args["batch_size"], 
                            shuffle=False, 
                            collate_fn=collate_with_pad3, 
                            num_workers=max(2, args["num_workers"]//2),
                            pin_memory=args.get("pin_memory", True),
                            persistent_workers=False, # 验证可以不用 persistent workers
                            )

    #4.Model
    model = TransformerModel(vocab_size, vocab_size, args["num_layers"],
                             args["d_model"], args["d_ff"], args["heads"],
                             args["dropout"], max_len=args["max_len"]).to(args["device"])
    criterion = LabelSmoothingloss(args["label_smoothing"], vocab_size, ignore_index=3)
    #加入weight_decay防止过拟合
    optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-5)
    scheduler = NoamOpt(args["d_model"], 1, args["warmup"], optimizer)

    best_bleu = 0.0 
    #早停参数
    patience = args.get("early_stop_patience", 5)
    min_delta = args.get("early_stop_min_delta", 0.0)
    no_improve = 0

    #在训练循环中加入AMP(自动混合精度，省显存)和GradScalar（防止在半精度下梯度下溢为0）
    scaler = torch.amp.GradScaler("cuda", enabled=args["device"].startswith("cuda"))
    for epoch in range(1, args["epochs"] + 1):
        model.train()
        total_loss = 0.0
        total_tokens = 0
        for i, (src, tgt, src_mask, tgt_mask) in enumerate(train_loader):
            src = src.to(args["device"])
            tgt = tgt.to(args["device"])
            src_mask = src_mask.to(args["device"])
            tgt_mask = tgt_mask.to(args["device"])
            #shift target for input/gold
            tgt_input = tgt[:, :-1]
            tgt_gold = tgt[:, 1:]
            #creat tgt_mask for tgt_input
            size = tgt_input.size(1)
            subsequent = torch.triu(torch.ones((1, size, size), dtype=torch.uint8), diagonal=1)
            subsequent = (subsequent == 0).to(args["device"])
            tgt_mask2 = (tgt_input != 3).unsqueeze(-2) & subsequent

            with torch.amp.autocast("cuda", enabled=args["device"].startswith("cuda")):
                out = model(src, tgt_input, src_mask, tgt_mask2)#(batch, size, d_model)
                log_probs = model.generator(out) #(batch,size, vocab_size)
                loss = criterion(log_probs, tgt_gold) #(1,)
            #反向传播与梯度缩放
            scaler.scale(loss).backward()
            #先unscale再梯度裁剪
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            #update lr (Noam)
            scheduler.step() #仅更新学习率
            #执行optimizer.step via scaler
            scaler.step(optimizer)
            scaler.update()
            scheduler.zero_grad()

            total_loss += loss.item() * (tgt_gold != 3).sum().item()
            total_tokens += (tgt_gold != 3).sum().item()

            if (i + 1) % 100 == 0:
                print(f"Epoch {epoch} Step {i+1} Average {total_loss / total_tokens :.4f}")
        avg_loss = total_loss / total_tokens
        print(f"Epoch {epoch} Training loss per token: {avg_loss:.4f}")

        #验证（batch greedy search)
        model.eval()
        refs = []
        hyps = []
        with torch.no_grad():
            for src, tgt, src_mask, tgt_mask in val_loader:
                src = src.to(args["device"])
                src_mask = src_mask.to(args["device"])
                # # greedy decode : we will do per-example greedy for simplicity
                # for b in range(src.size(0)):
                #     src_b = src[b:b+1] #(1, max_len)
                #     src_mask_b = src_mask[b:b+1] #(1, max_len)
                #     ids = greedy_decode(model, sp, src_b, src_mask_b, args["max_decoding_len"], args["device"]).squeeze(0).cpu().numpy().tolist()
                #     pred = ids_to_str(sp, ids)
                #     hyps.append(pred)
                #     tgt_ids = tgt[b].cpu().numpy().tolist()
                #     refs.append(ids_to_str(sp, tgt_ids))
                # batch greedy decode
                ys_batch = batched_greedy_decode(model, sp, src, src_mask,
                                                args["max_decoding_len"], args["device"],
                                                use_autocast=False)#(batch, max_len)
                #遍历batch 收集结果
                ys_batch = ys_batch.cpu().numpy().tolist()
                for ids, tgt_row in zip(ys_batch, tgt):
                    hyps.append(ids_to_str(sp, ids))
                    tgt_ids = tgt_row.cpu().numpy().tolist()
                    refs.append(ids_to_str(sp, tgt_ids)) 
        #compute BLEU
        bleu = sacrebleu.corpus_bleu(hyps, [refs])
        print(f"Epoch {epoch} BLEU: {bleu.score:.2f}")

        #保存并early_stop判断
        if bleu.score > best_bleu + min_delta:
            best_bleu = bleu.score
            no_improve = 0
            torch.save({
                "model": model.state_dict(),
                "sp_model": args["sp_model"],
                "vocab_size": vocab_size,
                "args": args
            }, os.path.join(args["save_dir"], "best_model.pt"))
            print("Saved best model, BLEU:", best_bleu)
        else:
            no_improve += 1
            print(f"No improvement for {no_improve}/{patience} epochs.")
        if no_improve >= patience:
            print(f"Early stopping triggered. No improvement in BLEU for {patience} consecutive epochs.")
            break
    print("Training finished. Best BLEU:", best_bleu)

    # #最终在测试集上做evaluation
    # #load best model
    # ckpt = torch.load(os.path.join(args["save_dir"], "best_model.pt"), map_location=args["device"])
    # model.load_state_dict(ckpt["model"])
    # model.to(args["device"])
    # model.eval()

    # #test evaluation greedy + beam
    # refs = []
    # hyps_greedy = []
    # hyps_beam = []
    # with torch.no_grad():
    #     for src, tgt, src_mask, tgt_mask in val_loader:
    #         src = src.to(args["device"])
    #         src_mask = src_mask.to(args["device"])
    #         for b in range(src.size(0)):
    #             src_b = src[b:b+1]
    #             src_mask_b = src_mask[b:b+1]
    #             g_ids = greedy_decode(model, sp, src_b, src_mask_b, args["max_decoding_len"], args["device"]).squeeze(0).cpu().numpy().tolist()
    #             hyps_greedy.append(ids_to_str(sp, g_ids))
    #             b_ids = beam_search_decode(model, sp, src_b, src_mask_b, args["max_decoding_len"], args["beam_size"], args["device"]).squeeze(0).cpu().numpy().tolist()
    #             hyps_beam.append(ids_to_str(sp, b_ids))
    #             tgt_ids = tgt[b].cpu().numpy().tolist()
    #             refs.append(ids_to_str(sp, tgt_ids))
    #     bleu_g = sacrebleu.corpus_bleu(hyps_greedy, [refs])
    #     bleu_b = sacrebleu.corpus_bleu(hyps_beam, [refs])
    #     print("Final Greedy BLEU:", bleu_g.score)
    #     print("Final Beam BLEU:", bleu_b.score)


#测试代码，以及避免被别的文件导入
if __name__ == "__main__":
    #创建命令行参数解析器
    parser = argparse.ArgumentParser()
    #遍历默认配置字典，自动注册所有命令行参数
    for k, v in DEFAULTS.items():
        t = type(v)
        #如果默认值是None，强制把类型设为字符串
        if v is None:
            t = str
        #自动添加命令行参数： --参数名，类型，默认值
        parser.add_argument(f"--{k}", type=t, default=v)
    #解析命令行输入，转成字典
    parsed = vars(parser.parse_args())
    train_main(parsed)

    





