from pathlib import Path
import sys
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import modal
import torchaudio
import torch.nn as nn
import torchaudio.transforms as T
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR

from model import AudioCNN

app = modal.App("audio-CNN")

image = (modal.Image.debian_slim()
         .pip_install_from_requirements("requirements.txt")
         .apt_install(["wget", "unzip", "ffmpeg", "libsndfile1"])
         .run_commands([
             "cd /tmp && wget https://github.com/karolpiczak/ESC-50/archive/master.zip -O esc50.zip",
             "cd /tmp && unzip esc50.zip",
             "mkdir -p /opt/esc50-data",
             "cp -r /tmp/ESC-50-master/* /opt/esc50-data/",
             "rm -rf /tmp/esc50.zip /tmp/ESC-50-master"
         ])
         .add_local_python_source("model"))

volume = modal.Volume.from_name("esc50-data", create_if_missing=True) # before training
model_volume = modal.Volume.from_name("esc-modal", create_if_missing=True) # after training

class ESC50Dataset(Dataset):
    def __init__(self, data_dir, metadata_file, split="train", transform=None):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.metadata = pd.read_csv(metadata_file)
        self.split = split
        self.transform = transform
        
        if split == 'train':
            self.metadata = self.metadata[self.metadata['fold'] != 5]
        else:
            self.metadata = self.metadata[self.metadata['fold'] == 5]
        
        self.classes = sorted(self.metadata['category'].unique())
        self.class_to_index = {cls: idx for idx, cls in enumerate(self.classes)}
        self.metadata['label'] = self.metadata['category'].map(self.class_to_index)
        
    def __len__(self):
        return len(self.metadata)   
    
    def __getitem__(self, index):
        row = self.metadata.iloc[index]
        audio_path = self.data_dir / "audio" / row['filename']  # datadir/audio/1-100032-A-0.wav
        
        waveform, sample_rate = torchaudio.load(audio_path)
        #waveform ==> [channels, samples] = [2, 440955]
        if waveform.shape[0] > 1: # [channels, samples] = [2, 440955] --> [1, 440955]
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        if self.transform:
            spectogram = self.transform(waveform)
        else:
            spectogram = waveform
        
        return spectogram, row['label']

@app.function(image=image, gpu="T4", volumes={"/data": volume, "/models": model_volume}, timeout=60*60*2)
def train():
    esc50_dir = Path("/opt/esc50-data")
    train_transform = nn.Sequential(
        T.MelSpectrogram(
            sample_rate= 22050, n_fft=1024,
            hop_length=512, n_mels=128,
            f_min=0, f_max=11025
        ),
        T.AmplitudeToDB(),
        T.FrequencyMasking(freq_mask_param=30),
        T.TimeMasking(time_mask_param=80)
    )  
    
    val_transform = nn.Sequential(
        T.MelSpectrogram(
            sample_rate= 22050, n_fft=1024,
            hop_length=512, n_mels=128,
            f_min=0, f_max=11025
        ),
        T.AmplitudeToDB(),
    )  
    
    train_dataset = ESC50Dataset(data_dir=esc50_dir, metadata_file=esc50_dir / "meta" / "esc50.csv", split="train", transform=train_transform)
        
    val_dataset = ESC50Dataset(data_dir=esc50_dir, metadata_file=esc50_dir / "meta" / "esc50.csv", split="val", transform=val_transform)
    
    print(f"Training Samples: {len(train_dataset)}")
    print(f"Validation Samples: {len(val_dataset)}") 
    
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True) # To randomize the order of training data
    test_loader = DataLoader(val_dataset, batch_size=32, shuffle=False) # To maintain the original sequence of Test Data
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AudioCNN(num_classes=len(train_dataset.classes))
    model.to(device)
    
    num_epochs = 100
    # label_smoothing helps the model to learn to have more reasonable predictions.
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1) # [1, 0, 0, 0, 0] -> [0.9, 0.025, 0.025. 0.025, 0.025] 
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01) # lr -> learning rates - how fast the model learns, weight_decay -> parameter to prevent overfitting by penalizing large weights.
    
    scheduler = OneCycleLR(
        optimizer, max_lr=0.002, epochs=num_epochs,
        steps_per_epoch=len(train_dataloader), # no. of optimizing steps hapenning per epoch
        pct_start=0.1 # This means that it spends 10% of training increasing learning rate and 90% of training decreasing the learning rate 
    )
        
@app.local_entrypoint()
def main():
    train.remote()
