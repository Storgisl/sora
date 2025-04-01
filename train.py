import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW
from model import ConditionalMusicGPT

# Load tokenized dataset
with open("tokenized_dataset.json", "r") as f:
    tokenized = json.load(f)

# Create mappings
genres = list(set([t[0].split("=")[1] for t in tokenized))
instruments = list(set([tok.split("=")[1] for t in tokenized for tok in t if "Instrument=" in tok))

class MIDIDataset(Dataset):
    def __init__(self, tokenized, genre_map, instr_map):
        self.tokenized = tokenized
        self.genre_map = genre_map
        self.instr_map = instr_map

    def __getitem__(self, idx):
        tokens = self.tokenized[idx]
        genre = self.genre_map[tokens[0].split("=")[1]]
        instruments = [self.instr_map[tok.split("=")[1]] for tok in tokens[1:] if "Instrument=" in tok]
        # Pad instruments to 2
        instruments = instruments[:2] + [-1]*(2 - len(instruments))
        return {
            "input_ids": tokens[3:],  # Skip genre/instrument tokens
            "genre_ids": genre,
            "instr_ids": instruments
        }

dataset = MIDIDataset(tokenized, {g:i for i,g in enumerate(genres)}, {i:j for j,i in enumerate(instruments)})
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Initialize model
model = ConditionalMusicGPT(num_genres=len(genres), num_instruments=len(instruments))
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training loop
for epoch in range(5):
    for batch in dataloader:
        input_ids = batch["input_ids"]
        genre_ids = batch["genre_ids"]
        instr_ids = batch["instr_ids"]
        
        outputs = model(input_ids, genre_ids, instr_ids)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"Loss: {loss.item()}")
