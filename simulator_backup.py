import curses
import torch
import pandas as pd

from utils import load_model

# Configuration & Mapping
KEY_MAP = {
    '1': 'ㄅ', 'q': 'ㄆ', 'a': 'ㄇ', 'z': 'ㄈ',
    '2': 'ㄉ', 'w': 'ㄊ', 's': 'ㄋ', 'x': 'ㄌ',
    'e': 'ㄍ', 'd': 'ㄎ', 'c': 'ㄏ',
    'r': 'ㄐ', 'f': 'ㄑ', 'v': 'ㄒ',
    '5': 'ㄓ', 't': 'ㄔ', 'g': 'ㄕ', 'b': 'ㄖ',
    'y': 'ㄗ', 'h': 'ㄘ', 'n': 'ㄙ',
    'u': 'ㄧ', 'j': 'ㄨ', 'm': 'ㄩ',
    '8': 'ㄚ', 'i': 'ㄛ', 'k': 'ㄜ', ',': 'ㄝ',
    '9': 'ㄞ', 'o': 'ㄟ', 'l': 'ㄠ', '.': 'ㄡ',
    '0': 'ㄢ', 'p': 'ㄣ', ';': 'ㄤ', '/': 'ㄥ',
    '-': 'ㄦ'
}

TONE_MAP = {
    '6': 'ˊ', '3': 'ˇ', '4': 'ˋ', '7': '˙', ' ': ' '
}

GROUPS = {
    "initial": "ㄅㄆㄇㄈㄉㄊㄋㄌㄍㄎㄏㄐㄑㄒㄓㄔㄕㄖㄗㄘㄙ",
    "medial": "ㄧㄨㄩ",
    "final": "ㄚㄛㄜㄝㄞㄟㄠㄡㄢㄣㄤㄥㄦ"
}

bopomofo_df = pd.read_csv("chars_zhuyin.csv")

bopomofo_dict = {} # zhuyin -> [chars...]
for ch, zy in zip(bopomofo_df["char"], bopomofo_df["zhuyin"]):
    bopomofo_dict.setdefault(zy, []).append(ch)


class IMESimulator:
    def __init__(self, model: torch.nn.Module, token2id: dict[str, int], id2token: dict[int, str]):
        self.model = model
        self.token2id = token2id
        self.id2token = id2token
        self.device = next(model.parameters()).device
        self.model.eval()
        
        self.committed_text = ""  # Text already entered
        self.current_composition = {"initial": "", "medial": "", "final": ""}
        self.candidates = []
        self.input_buffer = [] # History of committed tokens for context

    def get_zhuyin_string(self):
        return self.current_composition["initial"] + \
               self.current_composition["medial"] + \
               self.current_composition["final"]

    def process_key(self, char: str):
        # Handle Tone (Completion)
        if char in TONE_MAP:
            zhuyin = self.get_zhuyin_string()
            if zhuyin:
                tone = TONE_MAP[char]
                if tone == "˙":
                    zhuyin = tone + zhuyin
                elif tone != " ":
                    zhuyin = zhuyin + tone
                
                if zhuyin in bopomofo_dict:
                    self.infer_candidates(zhuyin)
                    return True
        
        # Handle Bopomofo input
        if char in KEY_MAP:
            bop = KEY_MAP[char]
            for group, members in GROUPS.items():
                if bop in members:
                    self.current_composition[group] = bop
                    self.candidates = []
                    return True
        
        # Handle Backspace
        if char == "BACKSPACE":
            if self.get_zhuyin_string():
                self.current_composition = {"initial": "", "medial": "", "final": ""}
            elif self.committed_text:
                self.committed_text = self.committed_text[:-1]
                if self.input_buffer: self.input_buffer.pop()
            return True
            
        return False

    def infer_candidates(self, zhuyin_syllable):
        """Run the model to get the most likely CJK character for the zhuyin."""
        with torch.no_grad():
            # Construct input: [Context...] + [Current Zhuyin]
            input_toks = self.input_buffer + [zhuyin_syllable]
            # Simple truncation for simulator
            input_toks = input_toks[-96:]
            
            ids = [self.token2id.get(t, self.token2id["[OTHER]"]) for t in input_toks]
            input_tensor = torch.tensor([ids]).to(self.device)
            
            logits = self.model(input_tensor, attention_mask=torch.ones_like(input_tensor))
            
            # Get top k predictions for the last position
            probs = torch.softmax(logits[0, -1], dim=-1)
            top_indices = torch.argsort(probs, dim=-1, descending=True)

            # Filter valid characters
            valid_chars = bopomofo_dict[zhuyin_syllable]
            
            self.candidates = []
            for idx in top_indices:
                char = self.id2token[idx.item()]
                # Only suggest CJK characters (skip Zhuyin/Special tokens)
                if '\u4e00' <= char <= '\u9fff' and char in valid_chars:
                    self.candidates.append(char)
            
            if not self.candidates:
                self.candidates = ["?"]

    def commit_candidate(self, idx):
        if 0 <= idx < len(self.candidates):
            chosen = self.candidates[idx]
            self.committed_text += chosen
            self.input_buffer.append(chosen)
            self.current_composition = {"initial": "", "medial": "", "final": ""}
            self.candidates = []

def main(stdscr: curses.window):
    # Initialize UI
    curses.curs_set(0)
    stdscr.nodelay(False)
    
    # Load your model
    model, token2id, id2token = load_model("model_allbop_8l_512d.pt", torch.device("cpu"))
    simulator = IMESimulator(model, token2id, id2token)

    while True:
        stdscr.clear()
        stdscr.addstr(1, 2, "=== IME Model Simulator (Press 'ESC' to quit) ===")
        
        # Display Text Area
        stdscr.addstr(3, 2, f"Text: {simulator.committed_text}", curses.A_BOLD)
        
        # Display Composition Bar
        comp = simulator.get_zhuyin_string()
        stdscr.addstr(5, 2, f"Composition: [{comp}]", curses.color_pair(0))
        
        # Display Candidates
        if simulator.candidates:
            cand_str = "  ".join([f"{i+1}:{c}" for i, c in enumerate(simulator.candidates[:9])])
            stdscr.addstr(7, 2, f"Candidates: {cand_str}", curses.A_REVERSE)

        stdscr.refresh()

        key = stdscr.getch()
        
        if key == 27: # ESC
            break
        elif key == curses.KEY_BACKSPACE or key == 127:
            simulator.process_key("BACKSPACE")
        elif ord('1') <= key <= ord('9') and simulator.candidates:
            simulator.commit_candidate(int(chr(key)) - 1)
        else:
            try:
                char = chr(key)
                simulator.process_key(char)
            except:
                pass


curses.wrapper(main)
