import base64
import io
import requests
import librosa
import numpy as np
import modal
import torch.nn as nn
import torchaudio.transforms as T
import torch
import soundfile as sf
from pydantic import BaseModel

from model import AudioCNN

app = modal.App("Audio_CNN_Inference")

image = (modal.Image.debian_slim()
         .pip_install_from_requirements("requirements.txt")
         .apt_install(['libsndfile1'])
         .add_local_python_source("model"))

model_volume = modal.Volume.from_name("esc-modal") # after training

class AudioProcessor:
    def __init__(self):
        self.transform = nn.Sequential(
            T.MelSpectrogram(
             sample_rate= 22050, n_fft=1024,
             hop_length=512, n_mels=128,
             f_min=0, f_max=11025
            ),
            T.AmplitudeToDB()
        )
        
    def process_audio_chunk(self, audio_data): # to convert the audio data into an understandable data
        waveform = torch.from_numpy(audio_data).float()
        
        waveform = waveform.unsqueeze(0) # This adds another channel on a dimension of Tensor
        
        spectogram = self.transform(waveform)
        
        return spectogram.unsqueeze(0)

class InferenceRequest(BaseModel):
    audio_data: str

# Inference Endpoint    
@app.cls(image=image, gpu="T4", volumes={"/models": model_volume}, scaledown_window=15)    
class AudioClassifier:
    @modal.enter()
    def load_model(self):
        print("Loading models on Enter.")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        checkpoint = torch.load('/models/best_model.pth', map_location=self.device)
        self.classes = checkpoint['classes']
        self.model = AudioCNN(num_classes=len(self.classes))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        self.audio_processor = AudioProcessor()
        print("Model loaded on Enter.")
    
    @modal.fastapi_endpoint(method="POST")    
    def inference(self, request: InferenceRequest):
        # production: frontend -> upload file to S3 -> Inference Endpoint -> Download from S3 Bucket
        # frontend -> send file directly -> Inference Endpoint 
        audio_files = base64.b64decode(request.audio_data)
        
        audio_data, sample_rates = sf.read(io.BytesIO(audio_files), dtype="float32")
        
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        if sample_rates != 22050:
            audio_data = librosa.resample(y=audio_data, orig_sr=sample_rates, target_sr=22050)
            
        spectogram = self.audio_processor.process_audio_chunk(audio_data)
        spectogram = spectogram.to(self.device)
        
        with torch.no_grad():
            output = self.model(spectogram)
            
            # Clean the model output
            output = torch.nan_to_num(output)
            probabilities = torch.softmax(output, dim=1) # dim=0 is batch, dim=1 is class (batch_sie, num_classes)
            
            # next we get the top 3 probabilities
            top3_probs, top3_indices = torch.topk(probabilities[0], 3)
            
            # next we create a dictionary of different classes and their confidences
            # dog: 0.9, chirping_birds: 0.2, etc
            # top3_probs: [0.9, 0.04, 0.5], top3_indices: [15, 42, 5]
            # zip function will create tuples like -> (0.9, 15), (0.04, 42), (0.5, 5)
            predictions = [{"class": self.classes[idx.item()], "confidence": prob.item()} for prob, idx in zip(top3_probs, top3_indices)]
            
        response = {
            "predictions": predictions
        }
        
        return response
    
@app.local_entrypoint()    
def main():
    audio_data, sample_rate = sf.read("dogbark.wav")
    
    buffer = io.BytesIO()
    sf.write(buffer, audio_data, 22050, format="WAV")
    # because we are sending JSON to the endpoint, and we can't send raw bytes in the json payload,so we need to convert it into bytes which can be sent as a regular string in the json
    audio_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    payload = {"audio_data": audio_b64}
    
    server = AudioClassifier()
    url = server.inference.get_web_url()
    response = requests.post(url= url, json=payload) # type: ignore
    response.raise_for_status()
    
    result = response.json()
    print("Top Predictions!!")
    for pred in result.get("predictions", []):
        print(f" -{pred["class"]} {pred["confidence"]:0.2%}")
      