
import re
import os
import math
import random
from collections import Counter
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

# 切换训练或推理：
# True -> 只训练（含验证、测试与保存最佳权重），不进入交互推理
# False -> 仅加载已保存的最佳权重，进入交互式推理
train_no_infer = False

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

MAX_LEN = 256
BATCH_SIZE = 64
LR = 5e-4
EPOCHS = 10
EMBED_DIM = 300
HIDDEN_DIM = 300
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5
MIN_FREQ = 2
MAX_VOCAB_SIZE = 50000

SPECIALS = ["<pad>", "<unk>"]
PAD, UNK = SPECIALS

_tok_re = re.compile(r"\w+|[^\w\s]", re.UNICODE)

CONTRACTIONS = {
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "can't": "can not",
    "couldn't": "could not",
    "shouldn't": "should not",
    "won't": "will not",
    "wouldn't": "would not",
    "hasn't": "has not",
    "haven't": "have not",
    "hadn't": "had not",
    "mustn't": "must not",
    "mightn't": "might not",
    "shan't": "shall not",
    "n't": " not",  # 兜底（少见残留）
}

NEGATORS = {"not", "no", "never"}
SENT_PUNCTS = {".", "!", "?"}
CONTRAST_WORDS = {"but", "however", "though", "although", "yet"}

def _apply_contrast_scope(tokens, scope=8):
    """遇到转折词后，将后续 scope 个词标注为 _CON"""
    out = []
    carry = 0
    for tok in tokens:
        if tok in SENT_PUNCTS:
            carry = 0
            out.append(tok)
            continue
        if tok in CONTRAST_WORDS:
            carry = scope
            out.append(tok + "_CONJ")  # 标记转折词自身
            continue
        if carry > 0 and tok.isalpha():
            out.append(tok + "_CON")
            carry -= 1
        else:
            out.append(tok)
    return out

def _normalize_contractions(text: str) -> str:
    text = text.lower()
    # 先替换长词形，最后用 n't 兜底
    for k, v in CONTRACTIONS.items():
        text = re.sub(rf"\b{k}\b", v, text)
    return text

def _apply_negation_scope(tokens, scope=3):
    """遇到否定词，将后续 scope 个词（遇句末标点提前终止）加 _NEG 后缀"""
    out = []
    carry = 0
    for tok in tokens:
        if tok in SENT_PUNCTS:
            carry = 0
            out.append(tok)
            continue
        if tok in NEGATORS:
            carry = scope
            out.append("not")  # 统一成 not
            continue
        if carry > 0 and tok.isalpha():
            out.append(tok + "_NEG")
            carry -= 1
        else:
            out.append(tok)
    return out

def tokenize(text: str) -> List[str]:
    text = text.lower()
    text = _normalize_contractions(text)
    tokens = _tok_re.findall(text)
    tokens = _apply_negation_scope(tokens)
    tokens = _apply_contrast_scope(tokens)
    return tokens

def build_vocab(texts: List[str], min_freq: int = 1, max_size: int = None) -> Tuple[dict, dict]:
    counter = Counter()
    for t in texts:
        counter.update(tokenize(t))

    # 过滤低频
    items = [(tok, freq) for tok, freq in counter.items() if freq >= min_freq]
    # 频率优先、再按 token 排序以稳定
    items.sort(key=lambda x: (-x[1], x[0]))

    if max_size is not None:
        items = items[: max(0, max_size - len(SPECIALS))]

    stoi = {PAD: 0, UNK: 1}
    for tok, _ in items:
        stoi[tok] = len(stoi)
    itos = {i: s for s, i in stoi.items()}
    return stoi, itos

def numericalize(text: str, stoi: dict, max_len: int = MAX_LEN) -> Tuple[List[int], int]:
    ids = [stoi.get(tok, stoi[UNK]) for tok in tokenize(text)]
    length = min(len(ids), max_len)
    if len(ids) < max_len:
        ids = ids + [stoi[PAD]] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return ids, length


class IMDBMapDataset(Dataset):
    def __init__(self, hf_split, stoi):
        self.data = hf_split
        self.stoi = stoi

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        label = int(item["label"])  # 0=neg, 1=pos
        ids, length = numericalize(text, self.stoi, MAX_LEN)
        return torch.tensor(label, dtype=torch.long), torch.tensor(ids, dtype=torch.long), torch.tensor(length, dtype=torch.long)


def collate_fn(batch):
    labels, ids, lengths = zip(*batch)
    labels = torch.stack(labels, 0)    # CPU 张量
    ids = torch.stack(ids, 0)          # CPU 张量
    lengths = torch.stack(lengths, 0)  # CPU 张量
    return labels, ids, lengths

from lstm import LSTMClassifier

def accuracy(logits, labels):
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean()

def run_epoch(dl, model, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    losses, accs = [], []
    for labels, ids, lengths in dl:
        # 在主进程搬到 GPU（避免 worker 初始化 CUDA）
        if device.type == "cuda":
            ids    = ids.pin_memory().to(device, non_blocking=True)
            labels = labels.pin_memory().to(device, non_blocking=True)
            # lengths 保持 CPU；如果你的 LSTMClassifier 内部已做 lengths.cpu()，也可以传 GPU
        else:
            ids = ids.to(device)
            labels = labels.to(device)

        logits = model(ids, lengths)          # 注意：lengths 传 CPU
        loss = criterion(logits, labels)
        acc = accuracy(logits, labels)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        losses.append(loss.item())
        accs.append(acc.item())
    return float(np.mean(losses)), float(np.mean(accs))


def train_and_eval():
    print("Loading IMDB...")
    dataset = load_dataset("imdb")  # train/test splits
    train_texts = [ex["text"] for ex in dataset["train"]]

    print("Building vocab...")
    stoi, itos = build_vocab(train_texts, min_freq=MIN_FREQ, max_size=MAX_VOCAB_SIZE)
    pad_idx = stoi[PAD]
    vocab_size = len(stoi)
    print(f"Vocab size: {vocab_size}")

    # 划分验证集（从训练集中抽取 5%）
    full_train = dataset["train"]
    n_total = len(full_train)
    n_valid = int(0.05 * n_total)
    # HuggingFace datasets 支持 shuffle + select
    full_train = full_train.shuffle(seed=SEED)
    valid_split = full_train.select(range(n_valid))
    train_split = full_train.select(range(n_valid, n_total))

    train_ds = IMDBMapDataset(train_split, stoi)
    valid_ds = IMDBMapDataset(valid_split, stoi)
    test_ds  = IMDBMapDataset(dataset["test"], stoi)
    
    NUM_WORKERS = 0  # 关键：避免在 Jupyter/conda 下 fork 后初始化 CUDA
    PIN_MEMORY = (device.type == "cuda")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )

    model = LSTMClassifier(
        vocab_size=vocab_size,
        embedding_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=2,
        n_layers=N_LAYERS,
        bidirectional=BIDIRECTIONAL,
        dropout=DROPOUT,
        pad_index=pad_idx,
    ).to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # === checkpoint dir ===
    ckpt_dir = os.path.join(os.path.dirname(__file__), "ckpts", "work3")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "lstm_imdb.pt")

    best_valid = math.inf
    print("Start training...")
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(train_loader, model, criterion, optimizer)
        va_loss, va_acc = run_epoch(valid_loader, model, criterion, optimizer=None)

        if va_loss < best_valid:
            best_valid = va_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "stoi": stoi,
                "config": {
                    "vocab_size": vocab_size,
                    "embedding_dim": EMBED_DIM,
                    "hidden_dim": HIDDEN_DIM,
                    "output_dim": 2,
                    "n_layers": N_LAYERS,
                    "bidirectional": BIDIRECTIONAL,
                    "dropout": DROPOUT,
                    "pad_index": pad_idx,
                    "max_len": MAX_LEN
                }
            }, ckpt_path)

        print(f"[Epoch {epoch}/{EPOCHS}] "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
              f"valid_loss={va_loss:.4f} valid_acc={va_acc:.4f}")

    # 测试集评估
    best = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(best["model_state_dict"]) 
    te_loss, te_acc = run_epoch(test_loader, model, criterion, optimizer=None)
    print(f"[Test] loss={te_loss:.4f} acc={te_acc:.4f}")

    print("Best checkpoint saved to:", ckpt_path)


def interactive_infer():
    # === checkpoint dir ===
    ckpt_dir = os.path.join(os.path.dirname(__file__), "ckpts", "work3")
    ckpt_path = os.path.join(ckpt_dir, "lstm_imdb.pt")
    if not os.path.exists(ckpt_path):
        print("Checkpoint not found:", ckpt_path)
        print("Please set train_no_infer=True and run training first.")
        return

    # load checkpoint
    bundle = torch.load(ckpt_path, map_location=device)
    cfg = bundle["config"]
    stoi = bundle["stoi"]

    # rebuild model from config
    model = LSTMClassifier(
        vocab_size=cfg["vocab_size"],
        embedding_dim=cfg["embedding_dim"],
        hidden_dim=cfg["hidden_dim"],
        output_dim=2,
        n_layers=cfg["n_layers"],
        bidirectional=cfg["bidirectional"],
        dropout=cfg["dropout"],
        pad_index=cfg["pad_index"],
    ).to(device)
    model.load_state_dict(bundle["model_state_dict"])  # type: ignore
    model.eval()

    id2label = {0: "neg", 1: "pos"}
    print("Loaded checkpoint:", ckpt_path)
    print("Enter a sentence to predict sentiment (q to quit):")
    with torch.no_grad():
        while True:
            try:
                text = input("> ").strip()
            except EOFError:
                break
            if text.lower() == "q":
                break
            if not text:
                continue
            ids, length = numericalize(text, stoi, cfg.get("max_len", MAX_LEN))
            ids_t = torch.tensor([ids], dtype=torch.long, device=device)
            len_t = torch.tensor([length], dtype=torch.long)
            logits = model(ids_t, len_t)
            probs = torch.softmax(logits, dim=-1).squeeze(0)
            pred = int(torch.argmax(probs).item())
            print(f"Pred: {id2label[pred]} | probs -> neg: {probs[0].item():.4f}, pos: {probs[1].item():.4f}")

if __name__ == "__main__":
    if train_no_infer:
        train_and_eval()
    else:
        interactive_infer()


