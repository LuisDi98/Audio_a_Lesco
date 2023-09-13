from flask import Flask, request, render_template
import torch
import whisper
import ipywidgets as widgets
from IPython import display as disp
from IPython.display import display, Audio, clear_output
import base64
from pydub import AudioSegment
import io
import tempfile
import librosa
from scipy.io import wavfile
import numpy as np
import soundfile as sf
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import sounddevice as sd
from scipy.io.wavfile import write

app = Flask(__name__)
tokenizer = None
model = None




def transcript(file_name):
    text = model.transcribe(file_name)['text']
    print(text)
    return text

@app.route('/', methods=['GET', 'POST'])
def init():
    return render_template('index.html')

@app.route('/api/upload_file', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file.save(file.filename)
            text_result = transcript(file.filename)
            return render_template('index.html', result=text_result)
        else: print("No file")
    return render_template('index.html')

@app.route('/api/record_wav', methods=['GET', 'POST'])
def record_wav():
    fs = 44100  # Sample rate
    seconds = 10  # Duration of recording
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    sd.wait()  # Wait until recording is finished
    file_name = 'output.wav'
    write(file_name, fs, myrecording)  # Save as WAV file
    text_result = transcript(file_name)
    return render_template('index.html', result=text_result)

if __name__ == '__main__':
    model = whisper.load_model("tiny")
    app.run(debug=True)
