from flask import Flask, render_template, request, send_file
from faster_whisper import WhisperModel
import os
from datetime import datetime

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Modelo pequeno (rápido)
model = WhisperModel("tiny")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    file = request.files["file"]
    if file:
        filename = datetime.now().strftime("%Y%m%d_%H%M%S_") + file.filename
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        srt_path = filepath.rsplit(".", 1)[0] + ".srt"
        segments, _ = model.transcribe(filepath, language="en")

        with open(srt_path, "w", encoding="utf-8") as srt:
            for i, seg in enumerate(segments, start=1):
                start = seg.start
                end = seg.end
                text = seg.text.strip()
                srt.write(f"{i}\n")
                srt.write(f"{int(start//3600):02}:{int((start%3600)//60):02}:{int(start%60):02},000 --> "
                          f"{int(end//3600):02}:{int((end%3600)//60):02}:{int(end%60):02},000\n")
                srt.write(f"{text}\n\n")

        return send_file(srt_path, as_attachment=True)

    return "Nenhum arquivo enviado."

if __name__ == "__main__":
    app.run(debug=True)
