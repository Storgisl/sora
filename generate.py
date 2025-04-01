from model import ConditionalMusicGPT
import torch

# Load trained model
model = ConditionalMusicGPT(num_genres=5, num_instruments=10)
model.load_state_dict(torch.load("model.pth"))

# Generate jazz piano sequence
genre_id = 1  # Jazz
instr_ids = [3, 7]  # Piano, drums
input_ids = [model.config.bos_token_id]  # Start token

for _ in range(200):  # Generate 200 tokens
    outputs = model(
        torch.tensor([input_ids]), torch.tensor([genre_id]), torch.tensor([instr_ids])
    )
    next_token = torch.argmax(outputs.logits[0, -1]).item()
    input_ids.append(next_token)

# Decode tokens to MIDI
from miditok import REMI

tokenizer = REMI()
midi = tokenizer.tokens_to_midi(input_ids)
midi.dump("generated.mid")
