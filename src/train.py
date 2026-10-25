import os
import shutil
from tqdm import tqdm
import random
import gc
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import pandas as pd
from datasets import load_dataset

from model import IMEModel

def setup():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

setup()
local_rank = int(os.environ["LOCAL_RANK"])
device = torch.device(f"cuda:{local_rank}")

CJK_START = '\u4e00'
CJK_END   = '\u9fff'
PUNCS = set("，。！？；：、（）「」『』")

MAX_CONTEXT = 96
MAX_TAIL = 8

# Model hyperparameters
num_layers = 8
hidden_size = 512
num_heads = 8
ffn_size = 2048
max_len = 96

# Training hyperparameters
learning_rate = 1e-3
lr_warmup_steps = 650
lr_decay = 1

batch_size = 128
accum_steps = 1
num_epochs = 100 #150
log_interval = 30
save_interval = 3

bopomofo_df = pd.read_csv("chars_zhuyin.csv")

bopomofo_dict = {}
for ch, zy in zip(bopomofo_df["char"], bopomofo_df["zhuyin"]):
    bopomofo_dict.setdefault(ch, []).append(zy)

MODEL_FILE = "ime_checkpoint.pt"
CONTINUE_TRAINING = False
checkpoint = None


def is_cjk(ch):
    return CJK_START <= ch <= CJK_END


def is_punc(ch):
    return ch in PUNCS


def is_zhuyin_token(tok):
    # 注音字元範圍：ㄅ(U+3105) ~ ㄦ(U+3126)
    return any('\u3105' <= c <= '\u3126' for c in tok)


def load_word_dict(file_path):
    df = pd.read_csv(file_path, comment='#', header=None, names=['word', 'priority', 'zhuyin'])
    word_to_data = {}
    for _, row in df.iterrows():
        w = str(row['word'])
        p = int(row['priority'])
        z = str(row['zhuyin']).split()
        
        # Ensure multi-char words have matching Zhuyin counts
        if len(w) > 1 and len(z) != len(w):
            continue 
            
        # Keep the entry with highest priority for each word
        if w not in word_to_data or p > word_to_data[w][0]:
            word_to_data[w] = (p, z)
    return word_to_data


WORD_DICT = load_word_dict("words_zhuyin.csv")


def get_best_zhuyin(window, index, word_to_data):
    """Finds the correct Bopomofo for window[index] using word context."""
    best_len = -1
    best_priority = -1
    best_zhuyin = None
    
    # Check words from length 11 down to 1 (Longest Match)
    for length in range(11, 0, -1):
        for start in range(index - length + 1, index + 1):
            if start < 0 or start + length > len(window):
                continue
            
            sub = window[start:start+length]
            if sub in word_to_data:
                priority, zhuyin_list = word_to_data[sub]
                if length > best_len or (length == best_len and priority > best_priority):
                    best_len = length
                    best_priority = priority
                    best_zhuyin = zhuyin_list[index - start]
        
        if best_len != -1: # Found the longest word
            return best_zhuyin
            
    return None


def generate_sample(text, word_to_data):
    total_len = MAX_CONTEXT + MAX_TAIL 
    end = random.randint(0, len(text) - 1)
    
    # 1. Determine Window (Clean Start vs Cross-sentence)
    lookback_limit = max(0, end - total_len + 1)
    actual_start = lookback_limit
    
    if random.random() < 0.50: # 50% chance to respect punctuation (Clean Start)
        for i in range(end - 1, lookback_limit - 1, -1):
            if text[i] in "。！？":
                actual_start = i + 1
                break
    
    window = text[actual_start:end + 1]
    
    # 2. Identify Tail Positions
    tail_positions = []
    for i in range(len(window) - 1, -1, -1):
        ch = window[i]
        if is_cjk(ch) or is_punc(ch):
            tail_positions.append(i)
            if len(tail_positions) == MAX_TAIL: break
        else: break

    if not tail_positions: return None
    tail_positions = list(reversed(tail_positions))
    cjk_tail_indices = [idx for idx in tail_positions if is_cjk(window[idx])]
    if not cjk_tail_indices: return None

    # 3. 30/70 Replacement Strategy
    to_replace_indices = set()
    if False: #random.random() < 0.30:
        to_replace_indices = set(cjk_tail_indices)
    else:
        num_to_replace = random.randint(1, max(1, len(cjk_tail_indices)))
        to_replace_indices = set(random.sample(cjk_tail_indices, num_to_replace))

    # 4. Tokenization with Disambiguation
    input_tokens, target_tokens, loss_mask = [], [], []
    other_buffer = ""

    for i, ch in enumerate(window):
        if not is_cjk(ch) and not is_punc(ch):
            other_buffer += ch
            continue
        
        if other_buffer:
            input_tokens.append("[OTHER]"); target_tokens.append(None); loss_mask.append(False)
            other_buffer = ""

        if i in tail_positions:
            target_tokens.append(ch)
            if i in to_replace_indices:
                # USE DISAMBIGUATION HERE
                bop = get_best_zhuyin(window, i, word_to_data)
                if bop:
                    input_tokens.append(bop)
                    loss_mask.append(True)
                else: # Fallback
                    input_tokens.append(ch); loss_mask.append(False)
            else:
                input_tokens.append(ch); loss_mask.append(False)
        else:
            input_tokens.append(ch); target_tokens.append(None); loss_mask.append(False)

    if other_buffer:
        input_tokens.append("[OTHER]"); target_tokens.append(None); loss_mask.append(False)

    return input_tokens, target_tokens, loss_mask


def build_vocab(texts):
    vocab = set(["[PAD]", "[MASK]", "[OTHER]"])

    # Add all unique characters from your text corpus
    for text in tqdm(texts, disable=local_rank != 0):
        vocab.update([char for char in text if (is_cjk(char) or is_punc(char))])

    # Add all possible Bopomofo tokens from your dictionary
    for bop_list in bopomofo_dict.values():
        vocab.update(bop_list)

    return sorted(list(vocab))


# Load preprocessed wiki texts
texts = []
with open("preprocessed_wiki.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            texts.append(line)

vocab = []
token2id = {}
id2token = {}

if not CONTINUE_TRAINING:
    vocab = build_vocab(texts)
    token2id = {tok: i for i, tok in enumerate(vocab)}
    id2token = {i: tok for tok, i in token2id.items()}


class IMEDataset(Dataset):
    def __init__(self, texts, word_dict, token2id, max_len):
        self.texts = texts
        self.word_dict = word_dict
        self.token2id = token2id
        self.max_len = max_len
        self.pad_id = token2id["[PAD]"]
        self.other_id = token2id["[OTHER]"]

    def __len__(self):
        # You can define an "epoch" as one pass through the raw lines
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        # 1. Generate the random mask/zhuyin sample for this specific line
        sample = generate_sample(text, self.word_dict)

        # 2. If the line was too short or invalid, pick a random one until we get a hit
        while sample is None:
            text = random.choice(self.texts)
            sample = generate_sample(text, self.word_dict)

        input_tokens, target_tokens, loss_mask = sample

        # 3. Convert to IDs (same logic as before)
        input_ids = [self.token2id.get(t, self.other_id) for t in input_tokens]
        target_ids = []
        for tok, m in zip(target_tokens, loss_mask):
            target_ids.append(self.token2id.get(tok, self.other_id) if m else -100)

        # 4. Padding / Truncation
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


def save_checkpoint(model, optimizer, scheduler, global_step):
    if local_rank != 0:
        return

    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
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


if CONTINUE_TRAINING and os.path.exists(MODEL_FILE):
    if local_rank == 0:
        print(f"Loading checkpoint from {MODEL_FILE} ...")
    checkpoint = torch.load(MODEL_FILE, map_location="cpu")

model = IMEModel(
    vocab_size=len(vocab),
    hidden_size=hidden_size,
    num_layers=num_layers,
    num_heads=num_heads,
    ffn_size=ffn_size,
    max_len=max_len,
)
if checkpoint:
    model = IMEModel(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
model.to(device)
model = DDP(model, device_ids=[local_rank])

criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
if checkpoint:
    optimizer.load_state_dict(checkpoint["optimizer_state"])

warmup_s = optim.lr_scheduler.LinearLR(optimizer, start_factor=1 if CONTINUE_TRAINING else 1e-2, end_factor=1.0, total_iters=lr_warmup_steps)
#decay_s = optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)
decay_s = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=lr_decay)
scheduler = optim.lr_scheduler.SequentialLR(optimizer, [warmup_s, decay_s], milestones=[lr_warmup_steps])
if checkpoint:
    scheduler.load_state_dict(checkpoint["scheduler_state"])

global_step = 0
if checkpoint:
    token2id = checkpoint["token2id"]
    id2token = checkpoint["id2token"]
    vocab = list(token2id.keys())
    global_step = checkpoint["step"]

if local_rank == 0:
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Sample vocabulary: {random.sample(vocab, 20)}")

dataset = IMEDataset(texts, WORD_DICT, token2id, max_len=max_len)
sampler = DistributedSampler(dataset)
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False,
    sampler=sampler,
    #persistent_workers=True,
    num_workers=0
)

model.train()

tbWriter = None
if local_rank == 0:
    if os.path.exists("logs/train") and not CONTINUE_TRAINING:
        shutil.rmtree("logs/train", ignore_errors=True)
    tbWriter = SummaryWriter("logs/train")

# Accuracy statistics
correct_all = 0
total_all = 0

for epoch in range(num_epochs):
    if local_rank == 0:
        print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")

    for batch in tqdm(dataloader):
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        loss = criterion(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1),
        )

        scaled_loss = loss / accum_steps
        scaled_loss.backward()

        if (global_step + 1) % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        scheduler.step()

        # Accuracy calculation
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

        if local_rank == 0 and global_step % log_interval == 0:
            acc_all = correct_all / max(1, total_all)

            tqdm.write(
                f"step {global_step:6d} | "
                f"lr {scheduler.get_last_lr()[0]:.2e} | "
                f"loss {loss.item():.4f} | "
                f"acc {acc_all:.3f} | "
            )

            tbWriter.add_scalar("Accuracy", acc_all, global_step)

            # Reset statistics for next log
            correct_all = total_all = 0

        if local_rank == 0 and global_step % 60 == 0:
            decode_prediction(
                input_ids.detach().cpu(),
                logits.detach().cpu(),
                target_ids.detach().cpu(),
                id2token,
            )

        if local_rank == 0:
            tbWriter.add_scalar("Loss", loss.item(), global_step)
            tbWriter.add_scalar("Learning Rate", scheduler.get_last_lr()[0], global_step)
        
        if (global_step + 1) % 200 == 0:
            gc.collect()

        global_step += 1
    
    if local_rank == 0 and (epoch + 1) % save_interval == 0:
        save_checkpoint(model.module, optimizer, scheduler, global_step)
    
    gc.collect()

if local_rank == 0:
    tbWriter.close()
    save_checkpoint(model.module, optimizer, scheduler, global_step)
    print(f"Model saved to {MODEL_FILE}")

torch.distributed.destroy_process_group()
