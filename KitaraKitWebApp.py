import streamlit as st
import pandas as pd
from io import StringIO
import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision import datasets
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch
import torchaudio
import numpy as np
import random
import librosa
import time
import random as rand

effect_to_int = {
    "modulation": 0,
    "reverb": 1,
    "delay": 2,
    "distortion": 3,
    "clean": 4,
    "distortion-delay": 5,
    "distortion-reverb": 6,
    "distortion-modulation": 7,
    "delay-reverb": 8,
    "delay-modulation": 9,
    "reverb-modulation": 10,
    "distortion-delay-reverb": 11,
    "distortion-delay-modulation": 12,
    "distortion-reverb-modulation": 13,
    "delay-reverb-modulation": 14,
    "distortion-delay-reverb-modulation": 15
}

in_channels = 1

class ConvNet1D(torch.nn.Module):
    def init(self):
        super().init()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, 3, 3),
            torch.nn.BatchNorm1d(3),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(3),
            torch.nn.Conv1d(3, 6, 3, stride = 2),
            torch.nn.BatchNorm1d(6),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(3),
            torch.nn.Conv1d(6, 3, 3, stride = 2),
            torch.nn.BatchNorm1d(3),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(3)
        )
        self.FC = torch.nn.Sequential(
            torch.nn.Linear(2664, 1200),
            torch.nn.ReLU(),
            torch.nn.Linear(1200, 16)
        )

    def forward(self, x):
      x = self.conv(x)
      x = x.flatten(start_dim=1)
      x = self.FC(x)

      return x

model = ConvNet1D()

model_path = "./best_model_found.pt"
model = torch.load(model_path)

st.title('KitarKit Demo')

uploaded_file = st.file_uploader("Choose a file", type=["wav", "mp3"])

percent_complete = 0
if uploaded_file is not None:

    audio_np, _ = librosa.load(uploaded_file, sr= None, mono = True)    
    st.audio(audio_np, sample_rate=48000)
    
    demo_audio = torch.tensor(audio_np)
    demo_audio = demo_audio.reshape(1, 1, 96000)

    result = None
    index = torch.argmax(model(demo_audio))
    for key, value in effect_to_int.items():
        if (value == index):
            result = key
            
    with st.container():    
        my_bar = st.progress(0)
        time.sleep(0.2)    

        for percent_complete in range(100):
            r = rand.random()
            n = rand.randint(1, 800)
            time.sleep(r/n)
            my_bar.progress(percent_complete + 1)
with st.container():        
    if(percent_complete == 99):
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        words = result.split("-")
        with col1:
            st.markdown("<h4 style='text-align: center;'>distortion</h4>", unsafe_allow_html=True)
            if "distortion" in words:
                st.image("./images/pedal_2.jpg")
            else:
                st.image("./images/pedal.jpg")
        with col2:
            st.markdown("<h4 style='text-align: center;'>reverb</h4>", unsafe_allow_html=True)
            if "reverb" in words:
                st.image("./images/pedal_1.jpg")
            else:
                st.image("./images/pedal.jpg")
        with col3:
            st.markdown("<h4 style='text-align: center;'>delay</h4>", unsafe_allow_html=True)
            if "delay" in words:
                st.image("./images/pedal.jpg")
            else:
                st.image("./images/pedal.jpeg")
        with col4:
            st.markdown("<h4 style='text-align: center;'>modulation</h4>", unsafe_allow_html=True)
            if "modulation" in words:
                st.image("./images/pedal_4.jpg")
            else:
                st.image("./images/pedal.jpg")
    else:
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        with col1:
            st.markdown("<h4 style='text-align: center;'>distortion</h4>", unsafe_allow_html=True)
            st.image("./images/pedal.jpeg")
        with col2:
            st.markdown("<h4 style='text-align: center;'>reverb</h4>", unsafe_allow_html=True)
            st.image("./images/pedal.jpeg")
        with col3:
            st.markdown("<h4 style='text-align: center;'>delay</h4>", unsafe_allow_html=True)
            st.image("./images/pedal.jpeg")
        with col4:
            st.markdown("<h4 style='text-align: center;'>modulation</h4>", unsafe_allow_html=True)
            st.image("./images/pedal.jpeg")