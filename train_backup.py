import os
from tqdm import tqdm
import random
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import pandas as pd

from model import IMEModel

CJK_START = '\u4e00'
CJK_END   = '\u9fff'
PUNCS = set("，。！？；：、（）「」『』")

MAX_CONTEXT = 96
MAX_TAIL = 8

BASE_MASK_PROB = 0.2
END_MASK_PROB  = 0.7
MASK_TOKEN = "[MASK]"

# Model hyperparameters
num_layers = 8
hidden_size = 512
num_heads = 8
ffn_size = 1024
max_len = 96

bopomofo_df = pd.read_csv("chars_zhuyin.csv")

bopomofo_dict = {}
for ch, zy in zip(bopomofo_df["char"], bopomofo_df["zhuyin"]):
    bopomofo_dict.setdefault(ch, []).append(zy)
 
MODEL_FILE = "ime_checkpoint.pt"
CONTINUE_TRAINING = True


def is_cjk(ch):
    return CJK_START <= ch <= CJK_END


def is_punc(ch):
    return ch in PUNCS


def is_zhuyin_token(tok):
    # 注音字元範圍：ㄅ(U+3105) ~ ㄦ(U+3126)
    return any('\u3105' <= c <= '\u3126' for c in tok)


def mask_probability(pos, tail_len):
    """
    pos: 0 = last char
    """
    if tail_len == 1:
        return BASE_MASK_PROB
    slope = (BASE_MASK_PROB - END_MASK_PROB) / (tail_len - 1)
    return BASE_MASK_PROB - slope * pos


def generate_sample(text, bopomofo_dict):
    """
    回傳:
      input_tokens: List[str]
      target_tokens: List[str]
      loss_mask: List[bool]  # True 表示這個位置要算 loss
    """

    if len(text) < MAX_TAIL + 5:
        return None

    # 1️⃣ 隨機選一個 tail 結束位置
    end = random.randint(MAX_TAIL, len(text) - 1)

    # 2️⃣ 取 context window
    start = max(0, end - MAX_CONTEXT + 1)
    window = text[start:end + 1]

    # 3️⃣ 從尾端往前找 tail
    tail_positions = []
    for i in range(len(window) - 1, -1, -1):
        ch = window[i]
        if is_cjk(ch) or is_punc(ch):
            tail_positions.append(i)
            if len(tail_positions) == MAX_TAIL:
                break
        else:
            break

    if not tail_positions:
        return None

    tail_positions = list(reversed(tail_positions))
    tail_len = len(tail_positions)

    input_tokens = []
    target_tokens = []
    loss_mask = []

    other_buffer = ""
    window_len = len(window)
    for i, ch in enumerate(window):
        is_tail = i in tail_positions
        cjk = is_cjk(ch)
        punc = is_punc(ch)
        last_char = (i == window_len - 1)

        if not cjk and not punc:
            # 非漢字非標點，累積
            other_buffer += ch
        # 遇到漢字、標點或尾端，處理 buffer
        if (cjk or punc or last_char):
            # 如果 buffer 有東西，先 flush
            if other_buffer:
                input_tokens.append("[OTHER]")
                target_tokens.append(None)
                loss_mask.append(False)
                other_buffer = ""
        if cjk or punc:
            if is_tail:
                tail_idx = tail_positions.index(i)
                if cjk:
                    p = mask_probability(tail_idx, tail_len)
                    r = random.random()
                    if r < p:
                        # 用注音或 MASK
                        bops = bopomofo_dict.get(ch)
                        if bops and random.random() < 0:
                            bop = random.choice(bops)
                            input_tokens.append(bop)
                        else:
                            input_tokens.append(MASK_TOKEN)
                        loss_mask.append(True)
                    else:
                        input_tokens.append(ch)
                        loss_mask.append(False) # 保留漢字，不算 loss
                    target_tokens.append(ch)
                else:
                    # 標點：保留，但不算 loss
                    input_tokens.append(ch)
                    target_tokens.append(ch)
                    loss_mask.append(False)
            else:
                input_tokens.append(ch)
                target_tokens.append(None)
                loss_mask.append(False)
        elif last_char and other_buffer:
            # 若最後一個字符是 other，且已 flush，在此分支不需要再處理
            pass

    return input_tokens, target_tokens, loss_mask


def generate_training_data(texts, bopomofo_dict, max_samples=100_000):
    samples = []
    attempts = 0

    while len(samples) < max_samples and attempts < max_samples * 5:
        text = random.choice(texts)
        sample = generate_sample(text, bopomofo_dict)
        if sample:
            samples.append(sample)
        attempts += 1

    return samples


def build_vocab(samples):
    vocab = set()
    for inp, tgt, _ in tqdm(samples):
        for tok in inp:
            vocab.add(tok)
        for tok in tgt:
            if tok is not None:
                vocab.add(tok)
    vocab.update(["[PAD]", "[MASK]", "[OTHER]"])
    return vocab


# --- Load preprocessed wiki texts ---
texts = []
with open("preprocessed_wiki.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            texts.append(line)

training_data = generate_training_data(texts, bopomofo_dict, max_samples=10_000)
del texts # free memory

vocab = build_vocab(training_data)
vocab = sorted(vocab)
token2id = {tok: i for i, tok in enumerate(vocab)}
id2token = {i: tok for tok, i in token2id.items()}

# --- PyTorch Dataset and DataLoader ---
class IMEDataset(Dataset):
    def __init__(self, samples: list[tuple], token2id: dict[str, int], max_len: int):
        self.samples = samples
        self.token2id = token2id
        self.max_len = max_len
        self.pad_id = token2id["[PAD]"]
        self.other_id = token2id["[OTHER]"]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_tokens, target_tokens, loss_mask = self.samples[idx]

        input_ids = [self.token2id.get(t, self.other_id) for t in input_tokens]
        target_ids = []
        for tok, m in zip(target_tokens, loss_mask):
            if m:
                target_ids.append(self.token2id.get(tok, self.other_id))
            else:
                target_ids.append(-100)

        # padding / truncation
        seq_len = len(input_ids)
        if seq_len < self.max_len:
            pad_len = self.max_len - seq_len
            input_ids = [self.pad_id] * pad_len + input_ids
            target_ids = [-100] * pad_len + target_ids
            attention_mask = [0] * pad_len + [1] * seq_len
        else:
            input_ids = input_ids[-self.max_len:]
            target_ids = target_ids[-self.max_len:]
            attention_mask = [1] * self.max_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


def decode_prediction(input_ids, logits, target_ids, id2token):
    preds = logits.argmax(dim=-1)

    for i in range(len(input_ids)):
        inp = [id2token[x.item()] for x in input_ids[i]]
        tgt = [
            id2token[x.item()] if x.item() != -100 else "_"
            for x in target_ids[i]
        ]
        out = [id2token[x.item()] for x in preds[i]]
        for pos in range(len(tgt)):
            if tgt[pos] == "_":
                out[pos] = "_"

        tqdm.write("IN : " + "".join(inp))
        tqdm.write("TGT: " + "".join(tgt))
        tqdm.write("OUT: " + "".join(out))
        tqdm.write("-" * 60)
        break


device = torch.device("mps" if torch.mps.is_available() else "cpu")
print("Using device:", device)

model = IMEModel(
    vocab_size=len(vocab),
    hidden_size=hidden_size,
    num_layers=num_layers,
    num_heads=num_heads,
    ffn_size=ffn_size,
    max_len=max_len,
).to(device)

criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
optimizer = optim.Adam(model.parameters(), lr=5e-5)
global_step = 0

if CONTINUE_TRAINING and os.path.exists(MODEL_FILE):
    print(f"Loading checkpoint from {MODEL_FILE} ...")
    checkpoint = torch.load(MODEL_FILE, map_location=device)
    model = IMEModel(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])

    if "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    token2id = checkpoint["token2id"]
    id2token = checkpoint["id2token"]
    vocab = list(token2id.keys())
    global_step = checkpoint["step"]

print(f"Vocabulary size: {len(vocab)}")
print(f"Sample vocabulary: {random.sample(vocab, 20)}")

dataset = IMEDataset(training_data, token2id, max_len=max_len)
dataloader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0
)

num_epochs = 1
scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1 if CONTINUE_TRAINING else 1e-2, total_iters=600)

log_interval = 50

model.train()

tbWriter = SummaryWriter("runs/train")

# --- Accuracy statistics ---
correct_all = 0
total_all = 0
correct_zhuyin = 0
total_zhuyin = 0
correct_mask = 0
total_mask = 0

for epoch in range(num_epochs):
    print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")

    for batch in tqdm(dataloader):
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        optimizer.zero_grad()

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        loss = criterion(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1),
        )

        # === Accuracy calculation ===
        with torch.no_grad():
            preds = logits.argmax(dim=-1)  # [B, L]
            for b in range(input_ids.size(0)):
                for t in range(input_ids.size(1)):
                    tgt = target_ids[b, t].item()
                    if tgt == -100:
                        continue
                    total_all += 1
                    if preds[b, t].item() == tgt:
                        correct_all += 1

                    inp_tok = id2token[input_ids[b, t].item()]
                    if inp_tok == MASK_TOKEN:
                        total_mask += 1
                        if preds[b, t].item() == tgt:
                            correct_mask += 1
                    elif is_zhuyin_token(inp_tok):
                        total_zhuyin += 1
                        if preds[b, t].item() == tgt:
                            correct_zhuyin += 1

        loss.backward()
        optimizer.step()
        scheduler.step()

        acc_all = correct_all / max(1, total_all)
        acc_z = correct_zhuyin / max(1, total_zhuyin)
        acc_m = correct_mask / max(1, total_mask)

        if global_step % log_interval == 0:
            tqdm.write(
                f"step {global_step:6d} | "
                f"loss {loss.item():.4f} | "
                f"acc {acc_all:.3f} | "
                f"zhuyin {acc_z:.3f} | "
                f"mask {acc_m:.3f}"
            )

            # Reset statistics for next log
            correct_all = total_all = 0
            correct_zhuyin = total_zhuyin = 0
            correct_mask = total_mask = 0

        if global_step % 100 == 0:
            decode_prediction(
                input_ids.cpu(),
                logits.cpu(),
                target_ids.cpu(),
                id2token,
            )
        
        tbWriter.add_scalar("Loss", loss.item(), global_step)
        tbWriter.add_scalar("Learning Rate", scheduler.get_last_lr()[0], global_step)
        tbWriter.add_scalar("Combined Accuracy", acc_all, global_step)
        tbWriter.add_scalar("Zhuyin Accuracy", acc_z, global_step)
        tbWriter.add_scalar("Mask Accuracy", acc_m, global_step)

        global_step += 1

tbWriter.close()
checkpoint = {
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "model_config": {
        "vocab_size": len(vocab),
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "ffn_size": ffn_size,
        "max_len": max_len,
    },
    "token2id": token2id,
    "id2token": id2token,
    "step": global_step,
}
torch.save(checkpoint, MODEL_FILE)

print(f"Model saved to {MODEL_FILE}")
