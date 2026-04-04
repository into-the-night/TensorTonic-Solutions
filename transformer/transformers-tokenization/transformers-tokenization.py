import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
        self.special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """

        for i, token in enumerate(self.special_tokens):
            self.word_to_id[token] = i
            self.id_to_word[i] = token

        self.vocab_size = len(self.special_tokens)

        for text in texts:
            words = text.split(" ")
            for word in words:
                if word not in self.word_to_id:
                    self.word_to_id[word] = self.vocab_size
                    self.id_to_word[self.vocab_size] = word 
                    self.vocab_size += 1
        

    def pad (self, ids: np.ndarray, max_length: int) -> List[int]:
        """
        Adds padding token id to a list of ids
        """
        return np.pad(ids, (0, np.max(max_length-len(ids), 0)), constant_values = self.word_to_id[self.pad_token])

    def encode(self, text: str) -> np.ndarray:
        """
        Convert text to array of token IDs.
        Use UNK for unknown words.
        """

        encoded = []
        
        for word in text.split(" "):
            if word in self.word_to_id:
                encoded.append(self.word_to_id[word])
            else:
                encoded.append(self.word_to_id[self.unk_token])

        return np.array(encoded)
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """

        decoded = []

        for id in ids:
            if id in self.id_to_word:
                decoded.append(self.id_to_word[id])

        return " ".join(decoded)