import torch
import soundfile as sf
import pickle
from synthesis import build_model
from synthesis import wavegen

id = 'ditte_test_22April04_1455_12'
spect_vc = pickle.load(open('results_'+id+'.pkl', 'rb'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = build_model().to(device)

for spect in spect_vc[24+1:]: # use [24:] to exclude p001, p002 and p003
    name = spect[0]
    c = spect[1]
    print(name)
    waveform = wavegen(model, c=c)   
    sf.write(name+'.wav', waveform, 16000)


