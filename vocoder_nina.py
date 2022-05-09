import torch
import soundfile as sf
import pickle
from synthesis import build_model
from synthesis import wavegen

id = 'chkpnt_stft_stft_scratch_22April21_1408_13_resumed_resumed_resumed'
model_type = 'stft'
spect_vc = pickle.load(open('/work3/dgro/VCTK-Corpus-0/stft/results_'+id+'.pkl', 'rb'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = build_model().to(device)

checkpoint = torch.load('checkpoint_step001000000_ema.pth', map_location=device)
model.load_state_dict(checkpoint["state_dict"])

for spect in spect_vc:
    name = spect[0]
    c = spect[1]
    print(c.shape)
    print(f'creating waveform from spect conversion: {name} refer to metadata.log for info')
    waveform = wavegen(model, c=c)   
    sf.write('/work3/dgro/VCTK-Corpus-0/'+str(model_type)+'/'+str(model_type)+'_'+name+'.wav', waveform, 16000)


