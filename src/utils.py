import torch
from model import IMEModel


def load_model(path, device):
    checkpoint = torch.load(path, map_location=device)

    model = IMEModel(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    token2id = checkpoint["token2id"]
    id2token = checkpoint["id2token"]

    return model, token2id, id2token
