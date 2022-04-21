import os
import pickle
import numpy as np
import soundfile as sf
from scipy import signal
from scipy.signal import get_window
from librosa.filters import mel
import librosa
from numpy.random import RandomState


class Spect(object):

    def __init__(self, config):
        """Initialize configurations."""

        self.speaker_embed = config.speaker_embed
        self.model_type = config.model_type
        self.targetDir = config.main_dir # Directory containing spectrograms
        self.cutoff = 30
        self.fs = 16000
        self.order = 5
        self.fft_length = 1024
        self.hop_length = 256
        self.n_fft = 1024
        self.n_mels = 128


    def butter_highpass(self):
        nyq = 0.5 * self.fs
        normal_cutoff = self.cutoff / nyq
        b, a = signal.butter(self.order, normal_cutoff, btype='high', analog=False)
        return b, a
    
    def pySTFT(self, x):
    
        x = np.pad(x, int(self.fft_length//2), mode='reflect')
    
        noverlap = self.fft_length - self.hop_length
        shape = x.shape[:-1]+((x.shape[-1]-noverlap)//self.hop_length, self.fft_length)
        strides = x.strides[:-1]+(self.hop_length*x.strides[-1], x.strides[-1])
        result = np.lib.stride_tricks.as_strided(x, shape=shape,
                                                strides=strides)
    
        fft_window = get_window('hann', self.fft_length, fftbins=True)
        result = np.abs(np.fft.rfft(fft_window * result, n=self.fft_length).T) # inverse function is irfft 
        return result
    
    def spect(self):
        mel_basis = mel(self.fs, self.n_fft, fmin=90, fmax=7600, n_mels=80).T
        min_level = np.exp(-100 / 20 * np.log(10))
        b, a = self.butter_highpass()
        
        # audio file directory
        rootDir = self.targetDir+'/wav48_silence_trimmed'
        saveDir = self.targetDir + '/' + self.model_type
        # specify if mic1 or mic2 should be used (mic2 is default)
        mic = 'mic1'

        dirName, subdirList, _ = next(os.walk(rootDir))
        #print('Found directory: %s' % dirName)

        for subdir in sorted(subdirList):
        #print(subdir)
            if not os.path.exists(os.path.join(saveDir, subdir)):
                os.makedirs(os.path.join(saveDir, subdir))
            _,_, fileList = next(os.walk(os.path.join(dirName,subdir)))
            prng = RandomState(int(subdir[1:])) 
            for fileName in sorted(fileList):
                if (mic in fileName) == False: # only use the specified mic
                    # Read audio file
                    x, fs = sf.read(os.path.join(dirName,subdir,fileName))
                    # Remove drifting noise
                    y = signal.filtfilt(b, a, x)
                    # Add a little random noise for model roubstness
                    wav = y * 0.96 + (prng.rand(y.shape[0])-0.5)*1e-06
                    # Compute spect
                    D = self.pySTFT(wav)
                    if self.model_type == 'spmel': # save mel spec
                        # Convert to mel and normalize
                        D_mel = np.dot(D.T, mel_basis)
                        D_db = 20 * np.log10(np.maximum(min_level, D_mel)) - 16
                        S = np.clip((D_db + 100) / 100, 0, 1)  
                    else: # save stft
                        S = D
                    # save spect
                    np.save(os.path.join(saveDir, subdir, fileName[:-5]), # -5 if flac files, -4 if wav files
                        S.astype(np.float32), allow_pickle=False)
