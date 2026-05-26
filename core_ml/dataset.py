import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

class WikiTextDataset(Dataset):
    def __init__(self, split="train", context_length=1024, max_tokens=None):
        """
        Loads the WikiText-2 dataset and tokenizes it.
        We use the 'gpt2' tokenizer for simplicity.
        """
        self.context_length = context_length
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        
        # Load a standard BPE tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Join all text and tokenize
        text = "\n".join(dataset["text"])
        if max_tokens: # Limit size for quick testing if needed
            text = text[:max_tokens * 4] # Rough approximation
            
        print(f"Tokenizing {split} split...")
        tokens = self.tokenizer(text, return_tensors="pt", truncation=False)["input_ids"].squeeze()
        
        # Truncate tokens to a multiple of context_length to form clean batches
        num_batches = len(tokens) // context_length
        self.tokens = tokens[:num_batches * context_length]
        
    def __len__(self):
        # We return sequences of size context_length
        # We subtract 1 because we need context_length + 1 tokens for the (x, y) pair
        return len(self.tokens) // self.context_length - 1

    def __getitem__(self, idx):
        # x is the sequence of tokens, y is the sequence shifted by 1 (next token prediction)
        start_idx = idx * self.context_length
        end_idx = start_idx + self.context_length
        
        x = self.tokens[start_idx:end_idx]
        y = self.tokens[start_idx+1:end_idx+1]
        return x, y

def get_dataloaders(batch_size=8, context_length=1024):
    train_dataset = WikiTextDataset(split="train", context_length=context_length)
    val_dataset = WikiTextDataset(split="validation", context_length=context_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, train_dataset.tokenizer.vocab_size
