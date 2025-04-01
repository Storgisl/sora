from miditok import REMI
from miditoolkit import MidiFile
import json

# Load labeled dataset
with open("midi_dataset.json", "r") as f:
    dataset = json.load(f)

# Initialize tokenizer
tokenizer = REMI(
    additional_tokens=["Genre=<GENRE>", "Instrument=<INSTRUMENT>"],
    params={"pitch_range": (21, 109), "beat_res": 4},
)

# Tokenize MIDIs and save
tokenized = []
for entry in dataset[:1000]:  # Start with 1k files for testing
    midi = MidiFile(entry["path"])
    tokens = tokenizer(midi)
    # Prepend genre/instrument tokens
    genre_token = f"Genre={entry['genre']}"
    instr_tokens = [
        f"Instrument={inst}" for inst in entry["instruments"][:2]
    ]  # Max 2 instruments
    full_tokens = [genre_token] + instr_tokens + tokens
    tokenized.append(full_tokens)

with open("tokenized_dataset.json", "w") as f:
    json.dump(tokenized, f)
