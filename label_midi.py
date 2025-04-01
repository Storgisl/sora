import os
from pretty_midi import PrettyMIDI
import json


def extract_labels(midi_path):
    try:
        midi = PrettyMIDI(midi_path)
        instruments = [
            midi.program_to_instrument_name(inst.program) for inst in midi.instruments
        ]
        # Mock genre classifier (replace with actual model)
        genre = "rock" if "rock" in midi_path.lower() else "classical"
        return {"genre": genre, "instruments": instruments}
    except:
        return None


dataset = []
for root, _, files in os.walk("lmd_full"):
    for file in files:
        if file.endswith(".mid"):
            midi_path = os.path.join(root, file)
            labels = extract_labels(midi_path)
            if labels:
                dataset.append({"path": midi_path, **labels})

with open("midi_dataset.json", "w") as f:
    json.dump(dataset, f)
