from flask import Flask, request, jsonify
import threading
from musegan_wrapper import run_musegan, read_progress

app = Flask(__name__)
generation_thread = None

@app.route("/generate", methods=["POST"])
def generate():
    global generation_thread

    if generation_thread and generation_thread.is_alive():
        return jsonify({"error": "Generation already in progress"}), 409

    data = request.get_json(force=True)
    genre = data.get("genre", "classical")
    tempo = data.get("tempo", 120)
    instruments = data.get("instruments", ["piano"])
    length_bars = data.get("length_bars", 4)

    def target():
        run_musegan(genre, tempo, instruments, length_bars)

    generation_thread = threading.Thread(target=target)
    generation_thread.start()

    return jsonify({"status": "generation started"})

@app.route("/progress", methods=["GET"])
def progress():
    return jsonify(read_progress())

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

