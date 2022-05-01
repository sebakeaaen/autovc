import torch
import soundfile as sf
import pickle
from synthesis import build_model
from synthesis import wavegen

id = 'chkpnt_spmel_reproducedAutoVC_new_22April23_1444_43_resumed_resumed_resumed_resumed'
model_type = 'spmel'
spect_vc = pickle.load(open('ninatest/results_'+id+'.pkl', 'rb'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = build_model().to(device)

checkpoint = torch.load('checkpoint_step001000000_ema.pth', map_location=device)
model.load_state_dict(checkpoint["state_dict"])

for spect in spect_vc:
    name = spect[0]
    c = spect[1]
    print(f'creating waveform from spect conversion: {name} refer to metadata.log for info')
    waveform = wavegen(model, c=c)   
    sf.write('ninatest/'+name+'.wav', waveform, 16000)


