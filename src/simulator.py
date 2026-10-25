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
    def __init__(self, model, token2id, id2token):
        self.model = model
        self.token2id = token2id
        self.id2token = id2token
        self.device = next(model.parameters()).device
        self.model.eval()
        
        self.committed_text = ""  
        self.current_syllable = {"initial": "", "medial": "", "final": ""}
        self.uncommitted_syllables = [] 
        self.uncommitted_chars = []     
        self.frozen_chars = {}          # {idx: 'CJK_char'}
        self.candidates = []            
        self.input_buffer = []          # List of committed CJK tokens
        
        self.cursor_idx = 0             
        self.selecting_mode = False

    def get_building_zhuyin(self):
        return self.current_syllable["initial"] + \
               self.current_syllable["medial"] + \
               self.current_syllable["final"]

    def process_key(self, char: str):
        if char in TONE_MAP:
            zhuyin = self.get_building_zhuyin()
            if zhuyin:
                tone = TONE_MAP[char]
                if tone == "˙": zhuyin = tone + zhuyin
                elif tone != " ": zhuyin = zhuyin + tone
                
                if zhuyin in bopomofo_dict:
                    self.uncommitted_syllables.insert(self.cursor_idx, zhuyin)
                    self.shift_frozen_indices(self.cursor_idx, 1)
                    self.current_syllable = {"initial": "", "medial": "", "final": ""}
                    self.cursor_idx += 1
                    self.update_predictions()
                    if len(self.uncommitted_syllables) > 8: self.auto_commit_leftmost()
                    return True

        if char in KEY_MAP:
            bop = KEY_MAP[char]
            for group, members in GROUPS.items():
                if bop in members:
                    self.current_syllable[group] = bop
                    return True

        if char == "BACKSPACE":
            if self.get_building_zhuyin():
                self.current_syllable = {"initial": "", "medial": "", "final": ""}
            elif self.uncommitted_syllables and self.cursor_idx > 0:
                idx_to_remove = self.cursor_idx - 1
                self.uncommitted_syllables.pop(idx_to_remove)
                if idx_to_remove in self.frozen_chars:
                    del self.frozen_chars[idx_to_remove]
                self.shift_frozen_indices(idx_to_remove, -1)
                
                self.cursor_idx -= 1
                self.update_predictions()
            return True
        return False

    def shift_frozen_indices(self, start_idx, delta):
        """Adjusts frozen character mapping when syllables are added/removed."""
        new_frozen = {}
        for idx, char in self.frozen_chars.items():
            if idx >= start_idx:
                if idx + delta >= 0:
                    new_frozen[idx + delta] = char
            else:
                new_frozen[idx] = char
        self.frozen_chars = new_frozen

    def update_predictions(self):
        """
        Best-First Greedy Decoding: 
        Repeatedly find the uncommitted syllable that has the highest 'max probability' 
        among its valid CJK candidates, freeze it, and re-run until all are decided.
        """
        if not self.uncommitted_syllables:
            self.uncommitted_chars = []
            return

        num_syllables = len(self.uncommitted_syllables)
        # temp_frozen stores the results of this decoding pass
        # We start with the user's manually frozen characters
        temp_frozen = self.frozen_chars.copy()
        
        # We need to fill exactly num_syllables
        while len(temp_frozen) < num_syllables:
            with torch.no_grad():
                # 1. Prepare mixed input based on current temp_frozen
                mixed_tokens = []
                for i, zy in enumerate(self.uncommitted_syllables):
                    if i in temp_frozen:
                        mixed_tokens.append(temp_frozen[i])
                    else:
                        mixed_tokens.append(zy)

                input_toks = (self.input_buffer + mixed_tokens)[-96:]
                ids = [self.token2id.get(t, self.token2id.get("[OTHER]", 0)) for t in input_toks]
                input_tensor = torch.tensor([ids]).to(self.device)

                # 2. Get model logits
                logits = self.model(input_tensor, attention_mask=torch.ones_like(input_tensor))
                uncommitted_logits = logits[0, -num_syllables:]

                best_overall_prob = -1.0
                best_pos = -1
                best_char_at_pos = "?"

                # 3. Search for the "most confident" undecided position
                for i in range(num_syllables):
                    if i in temp_frozen:
                        continue
                    
                    syllable = self.uncommitted_syllables[i]
                    valid_chars = bopomofo_dict.get(syllable, [])
                    if not valid_chars:
                        temp_frozen[i] = "?"
                        continue

                    probs = torch.softmax(uncommitted_logits[i], dim=-1)
                    
                    # Find the highest probability character that is actually valid for this Zhuyin
                    # This prevents the model from being "confident" about a character that doesn't match the Zhuyin
                    current_best_prob = -1.0
                    current_best_char = "?"
                    
                    for char in valid_chars:
                        char_id = self.token2id.get(char)
                        if char_id is not None:
                            p = probs[char_id].item()
                            if p > current_best_prob:
                                current_best_prob = p
                                current_best_char = char
                    
                    # Track which position across the whole sequence is the most certain
                    if current_best_prob > best_overall_prob:
                        best_overall_prob = current_best_prob
                        best_pos = i
                        best_char_at_pos = current_best_char

                # 4. Freeze the single most confident character and loop
                if best_pos != -1:
                    temp_frozen[best_pos] = best_char_at_pos
                else:
                    # Safety break if no valid chars found
                    for i in range(num_syllables):
                        if i not in temp_frozen: temp_frozen[i] = "?"
                    break

        # Finalize the order for display
        self.uncommitted_chars = [temp_frozen[i] for i in range(num_syllables)]
    
    def fetch_candidates_at_cursor(self):
        """Fetch candidates using the exact same mixed context."""
        idx = self.cursor_idx - 1
        if idx < 0: return

        mixed_tokens = []
        for i, zy in enumerate(self.uncommitted_syllables):
            if i in self.frozen_chars and i != idx: # Keep other frozen chars, but look for alternatives for this one
                mixed_tokens.append(self.frozen_chars[i])
            else:
                mixed_tokens.append(zy)

        input_toks = (self.input_buffer + mixed_tokens)[-96:]
        ids = [self.token2id.get(t, self.token2id.get("[OTHER]", 0)) for t in input_toks]
        input_tensor = torch.tensor([ids]).to(self.device)
        
        with torch.no_grad():
            logits = self.model(input_tensor, attention_mask=torch.ones_like(input_tensor))
            pos = - (len(self.uncommitted_syllables) - idx)
            probs = torch.softmax(logits[0, pos], dim=-1)
            top_indices = torch.argsort(probs, descending=True)
            
            valid_chars = bopomofo_dict.get(self.uncommitted_syllables[idx], [])
            self.candidates = [self.id2token[i.item()] for i in top_indices if self.id2token[i.item()] in valid_chars]
        
        self.selecting_mode = True

    def select_candidate(self, idx):
        if 0 <= idx < len(self.candidates):
            self.frozen_chars[self.cursor_idx - 1] = self.candidates[idx]
            self.selecting_mode = False
            self.update_predictions() # Global update based on new fixed character
    
    def commit_all(self):
        for char in self.uncommitted_chars:
            self.committed_text += char
            self.input_buffer.append(char)
        self.uncommitted_syllables = []
        self.uncommitted_chars = []
        self.frozen_chars = {}
        self.cursor_idx = 0
        self.selecting_mode = False

    def auto_commit_leftmost(self):
        char_to_commit = self.uncommitted_chars.pop(0)
        self.uncommitted_syllables.pop(0)
        # Clear frozen status for the char moving to permanent text
        if 0 in self.frozen_chars:
            del self.frozen_chars[0]
        self.shift_frozen_indices(1, -1)
        
        self.committed_text += char_to_commit
        self.input_buffer.append(char_to_commit)
        self.cursor_idx = max(0, self.cursor_idx - 1)
        self.update_predictions()

def main(stdscr: curses.window):
    curses.curs_set(0)
    curses.init_pair(1, curses.COLOR_CYAN, curses.COLOR_BLACK) # Uncommitted
    curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK) # Frozen/Locked
    
    model, token2id, id2token = load_model("ime_checkpoint.pt", torch.device("cpu"))
    simulator = IMESimulator(model, token2id, id2token)

    while True:
        stdscr.clear()
        stdscr.addstr(1, 2, "=== IME: Down to Select | Frozen chars influence context ===")
        stdscr.addstr(3, 2, "Text: ")
        stdscr.addstr(simulator.committed_text, curses.A_BOLD)
        
        for i, char in enumerate(simulator.uncommitted_chars):
            # Use a different color (Yellow) if the character is frozen
            color = curses.color_pair(2) if i in simulator.frozen_chars else curses.color_pair(1)
            style = color | curses.A_UNDERLINE
            if i == simulator.cursor_idx - 1:
                style |= curses.A_REVERSE
            stdscr.addstr(char, style)
            
        stdscr.addstr(5, 2, f"Building: [{simulator.get_building_zhuyin()}]")
        
        if simulator.selecting_mode and simulator.candidates:
            cand_str = "  ".join([f"{i+1}:{c}" for i, c in enumerate(simulator.candidates[:9])])
            stdscr.addstr(7, 2, f"[SELECTING] Pos {simulator.cursor_idx}:", curses.A_BOLD)
            stdscr.addstr(8, 2, cand_str, curses.A_REVERSE)

        stdscr.refresh()
        key = stdscr.getch()
        
        if key == 27: break
        elif key in [curses.KEY_BACKSPACE, 127, 8]:
            simulator.process_key("BACKSPACE")
        elif key == curses.KEY_LEFT:
            simulator.cursor_idx = max(0, simulator.cursor_idx - 1)
            simulator.selecting_mode = False
        elif key == curses.KEY_RIGHT:
            simulator.cursor_idx = min(len(simulator.uncommitted_syllables), simulator.cursor_idx + 1)
            simulator.selecting_mode = False
        elif key == curses.KEY_DOWN:
            simulator.fetch_candidates_at_cursor()
        elif key in [10, 13, curses.KEY_ENTER]:
            simulator.commit_all()
        elif simulator.selecting_mode and ord('1') <= key <= ord('9'):
            simulator.select_candidate(int(chr(key)) - 1)
        else:
            try:
                simulator.process_key(chr(key))
            except: pass

curses.wrapper(main)
