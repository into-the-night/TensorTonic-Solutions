from typing import List, Dict

class WordPieceTokenizer:
    """
    WordPiece tokenizer for BERT.
    """
    
    def __init__(self, vocab: Dict[str, int], unk_token: str = "[UNK]", max_word_len: int = 100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_word_len = max_word_len
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into WordPiece tokens.
        """
        tokens = []
        for word in text.lower().split():
            word_tokens = self._tokenize_word(word)
            tokens.extend(word_tokens)
        return tokens
    
    def _tokenize_word(self, word: str) -> List[str]:
        """
        Tokenize a single word into subwords.
        """
        if len(word) > self.max_word_len:
            return [self.unk_token]
            
        tokens = []
        token = ""
        i = len(word)
        j = 0

        while True:
            if j >= len(word):
                break

            t = word[:i] if j == 0 else "##"+word[j:i]

            if t in self.vocab:
                tokens.append(t)
                j = i
                i = len(word)
                
            else:
                if i >= j:
                    i -= 1
                else:
                    tokens = [self.unk_token]
                    break
            
        return tokens
