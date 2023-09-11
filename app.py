"""
1- Integrar wav2vec y probar en la pagina

2- 

"""


from flask import Flask, request, render_template
import torch
#from whisper import Whisper, ModelDimensions

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
    waveform, sample_rate = librosa.load(file_name, sr=44100)
    input_values = tokenizer(waveform, return_tensors='pt').input_values
    logits = model(input_values).logits
    preditcted_ids = torch.argmax(logits, dim=-1)
    text = tokenizer.batch_decode(preditcted_ids)[0]
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
    tokenizer = Wav2Vec2Tokenizer.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-spanish")
    model = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-spanish")
    # model_path = 'tiny_model.pth'
    # model = torch.hub.load('snakers4/silero-models', 'silero_whisper_tiny', model_path=model_path)
    app.run(debug=True)
