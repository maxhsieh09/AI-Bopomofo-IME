import torch
import time

from utils import load_model


def build_inference_input(
    context_tokens,
    tail_tokens,
    token2id,
    max_len,
):
    tokens = context_tokens + tail_tokens

    input_ids = [token2id.get(t, token2id["[OTHER]"]) for t in tokens]
    attention_mask = [1] * len(input_ids)

    if len(input_ids) < max_len:
        pad_len = max_len - len(input_ids)
        input_ids = [token2id["[PAD]"]] * pad_len + input_ids
        attention_mask = [0] * pad_len + attention_mask
    else:
        input_ids = input_ids[-max_len:]
        attention_mask = attention_mask[-max_len:]

    return (
        torch.tensor([input_ids], dtype=torch.long),
        torch.tensor([attention_mask], dtype=torch.long),
    )


def predict_tail(
    model,
    input_ids,
    attention_mask,
    id2token,
    tail_len,
    topk=5,
):
    with torch.no_grad():
        logits = model(input_ids, attention_mask)  # [1, L, vocab]

    probs = torch.softmax(logits, dim=-1)

    results = []
    for i in range(1, tail_len + 1):
        pos = -i
        p = probs[0, pos]

        values, indices = torch.topk(p, topk)
        candidates = [
            (id2token[idx.item()], values[j].item())
            for j, idx in enumerate(indices)
        ]
        results.append(list(candidates))

    return reversed(results)


device = torch.device("cpu") #torch.device("mps" if torch.mps.is_available() else "cpu")

model, token2id, id2token = load_model(
    "ime_checkpoint.pt",
    device,
)

context = []
tail = ["ㄓㄨˋ", "ㄧㄣ"]

input_ids, attention_mask = build_inference_input(
    context,
    tail,
    token2id,
    max_len=96,
)

candidates = predict_tail(
    model,
    input_ids.to(device),
    attention_mask.to(device),
    id2token,
    tail_len=len(tail),
    topk=5,
)

for i, cand in enumerate(candidates):
    print(f"Position {i}:")
    for ch, p in cand:
        print(f"  {ch} ({p:.3f})")
