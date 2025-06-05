import subprocess
import json
import os

status_path = "/tmp/musegan_progress.json"

def update_progress(stage, percent):
    with open(status_path, "w") as f:
        json.dump({"stage": stage, "percent": percent}, f)

def read_progress():
    if not os.path.exists(status_path):
        return {"stage": "not started", "percent": 0}
    try:
        with open(status_path, "r") as f:
            return json.load(f)
    except Exception:
        return {"stage": "error", "percent": 0}

import subprocess
from musegan_wrapper import update_progress

def run_musegan(genre, tempo, instruments, length_bars):
    update_progress("starting", 0)

    cmd = [
        "python", "/app/musegan/v1/main.py",
        "--genre", genre,
        "--tempo", str(tempo),
        "--length_bars", str(length_bars),
        "--instruments", ",".join(instruments),
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    while True:
        line = proc.stdout.readline()
        if not line:
            break
        print(line, end="")

        if "Progress:" in line:
            try:
                percent = int(line.split("Progress:")[1].strip().replace("%", ""))
                update_progress("generating", percent)
            except:
                pass

    proc.wait()
    if proc.returncode != 0:
        update_progress("failed", 0)
        return None

    update_progress("done", 100)

    # Wait for file to exist
    filename = f"gen_{genre}_{tempo}bpm_{length_bars}bars.mid"
    output_path = f"/app/musegan/v1/exps/temporal_hybrid/output/custom_generate/{filename}"

    timeout = 10
    while not os.path.exists(output_path) and timeout > 0:
        time.sleep(1)
        timeout -= 1

    if not os.path.exists(output_path):
        print("[ERROR] MIDI file was not found after generation.")
        return None

    return output_path

