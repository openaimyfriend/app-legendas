import os
from flask import Flask, render_template, request, send_file, abort
import soundfile as sf
from datetime import timedelta

os.environ.setdefault("CT2_FORCE_CPU_ISA", "GENERIC")

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

_model = None
def get_model():
    global _model
    if _model is None:
        # importa só quando for usar
        from faster_whisper import WhisperModel
        _model = WhisperModel(
            "tiny",
            device="cpu",
            compute_type="int8",
            cpu_threads=1
        )
    return _model

def format_srt_time(seconds: float) -> str:
    td = timedelta(milliseconds=int(seconds * 1000))
    total = int(td.total_seconds())
    ms = int(td.microseconds / 1000)
    hh = total // 3600
    mm = (total % 3600) // 60
    ss = total % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"

@app.route("/healthz")
def healthz():
    return "ok"

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        abort(400, "Arquivo não enviado")
    f = request.files["file"]
    if not f.filename:
        abort(400, "Arquivo inválido")

    in_path = os.path.join(app.config["UPLOAD_FOLDER"], f.filename)
    f.save(in_path)

    model = get_model()
    segments, info = model.transcribe(
        in_path,
        language="en",
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        beam_size=1
    )

    base = os.path.splitext(os.path.basename(in_path))[0]
    out_path = os.path.join(app.config["UPLOAD_FOLDER"], base + ".srt")

    with open(out_path, "w", encoding="utf-8") as srt:
        idx = 1
        for seg in segments:
            start = format_srt_time(seg.start)
            end = format_srt_time(seg.end)
            text = (seg.text or "").strip()
            if not text:
                continue
            srt.write(f"{idx}\n{start} --> {end}\n{text}\n\n")
            idx += 1

    return send_file(out_path, as_attachment=True, download_name=os.path.basename(out_path))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
