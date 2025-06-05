import os
from flask import Flask, request, jsonify, send_file
import threading
from musegan_wrapper import run_musegan, read_progress

app = Flask(__name__)
generation_thread = None
latest_output_path = None

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
        global latest_output_path
        latest_output_path = run_musegan(genre, tempo, instruments, length_bars)

    generation_thread = threading.Thread(target=target)
    generation_thread.start()

    return jsonify({
        "status": "generation started",
        "download_url": f"/download?genre={genre}&tempo={tempo}&length_bars={length_bars}"
    })


@app.route("/progress", methods=["GET"])
def progress():
    return jsonify(read_progress())

@app.route("/download", methods=["GET"])
def download():
    genre = request.args.get("genre")
    tempo = int(request.args.get("tempo"))
    length_bars = int(request.args.get("length_bars"))

    filename = f"gen_{genre}_{tempo}bpm_{length_bars}bars.mid"
    filepath = f"/app/musegan/v1/exps/temporal_hybrid/output/custom_generate/{filename}"

    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404

    return send_file(filepath, as_attachment=True, download_name=filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

