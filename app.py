import os
import uuid
import threading
from flask import Flask, render_template, request, jsonify, send_from_directory, abort
from faster_whisper import WhisperModel
import soundfile as sf

# ---------- Config ----------
UPLOAD_DIR = "uploads"
RESULT_DIR = "results"
MODEL_NAME = os.getenv("WHISPER_MODEL", "tiny")  # mude aqui se quiser
LANG = os.getenv("WHISPER_LANG", "en")           # "en" por padrão

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024  # até 1 GB por arquivo

# Estado dos jobs em memória: {job_id: {"pct": int, "status": str, "file": str}}
JOBS = {}

# Carrega o modelo uma vez (economiza MUITO tempo a cada requisição)
app.logger.info(f"Carregando modelo '{MODEL_NAME}'...")
model = WhisperModel(MODEL_NAME, device="cpu", compute_type="int8")  # int8 = leve/rápido em CPU
app.logger.info("Modelo pronto.")


def _transcrever_job(job_id: str, wav_path: str, total_dur: float):
    """Roda a transcrição em thread e atualiza progresso por tempo."""
    try:
        JOBS[job_id]["status"] = "Transcrevendo…"

        # Opções do faster-whisper
        segments, info = model.transcribe(
            wav_path,
            language=LANG,
            beam_size=1,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=300),
        )

        # Gera SRT enquanto atualiza %
        srt_path = os.path.join(RESULT_DIR, f"{job_id}.srt")
        idx = 1
        last_pct = 0

        def fmt(t):
            h = int(t // 3600)
            m = int((t % 3600) // 60)
            s = int(t % 60)
            ms = int(round((t - int(t)) * 1000))
            return f"{h:02}:{m:02}:{s:02},{ms:03}"

        with open(srt_path, "w", encoding="utf-8") as f:
            for seg in segments:
                start = max(0.0, seg.start)
                end = max(start, seg.end)
                # escreve bloco SRT
                f.write(f"{idx}\n{fmt(start)} --> {fmt(end)}\n{seg.text.strip()}\n\n")
                idx += 1

                # atualiza progresso pelo tempo decorrido
                if total_dur > 0:
                    pct = int(min(100, (end / total_dur) * 100))
                    if pct != last_pct:
                        JOBS[job_id]["pct"] = pct
                        last_pct = pct

        JOBS[job_id]["pct"] = 100
        JOBS[job_id]["status"] = "Pronto"
        JOBS[job_id]["file"] = os.path.basename(srt_path)

    except Exception as e:
        JOBS[job_id]["status"] = f"Erro: {e}"
        JOBS[job_id]["pct"] = 0


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/start", methods=["POST"])
def start():
    if "audio" not in request.files:
        return jsonify({"error": "arquivo não enviado"}), 400

    file = request.files["audio"]
    if file.filename == "":
        return jsonify({"error": "arquivo inválido"}), 400

    if not file.filename.lower().endswith(".wav"):
        return jsonify({"error": "envie um .wav"}), 400

    job_id = uuid.uuid4().hex
    wav_path = os.path.join(UPLOAD_DIR, f"{job_id}.wav")
    file.save(wav_path)

    # mede duração total para estimar %
    try:
        dur = float(sf.info(wav_path).duration)
    except Exception:
        dur = 0.0

    JOBS[job_id] = {"pct": 0, "status": "Iniciando…", "file": ""}

    # dispara em background
    th = threading.Thread(target=_transcrever_job, args=(job_id, wav_path, dur), daemon=True)
    th.start()

    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id):
    data = JOBS.get(job_id)
    if not data:
        return jsonify({"error": "job não encontrado"}), 404
    return jsonify(data)


@app.route("/download/<job_id>")
def download(job_id):
    data = JOBS.get(job_id)
    if not data or not data.get("file"):
        abort(404)
    return send_from_directory(RESULT_DIR, data["file"], as_attachment=True)


@app.route("/healthz")
def healthz():
    return "ok", 200


if __name__ == "__main__":
    # para rodar local: python app.py
    app.run(host="0.0.0.0", port=5000, debug=True)
