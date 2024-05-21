import os
import random
import torch
import torch.utils.data
import numpy as np
from scipy.io.wavfile import read
import math

MAX_WAV_VALUE = 32768.0

# Mel 필터 뱅크를 위한 유틸리티 함수
def librosa_mel_fn(sr, n_fft, n_mels, fmin, fmax):
    import librosa
    mel_basis = librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax)
    return mel_basis

def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate

def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)

def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C

def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output

def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output

mel_basis = {}
hann_window = {}

def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window

    if fmax not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    pad_amount = (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2))
    if pad_amount[0] > y.size(0):
        pad_amount = (y.size(0) - 1, y.size(0) - 1)

    # print("pad_amount", pad_amount)
    # print("before pad", y.size())
    y = torch.nn.functional.pad(y.unsqueeze(0), pad_amount, mode='reflect')
    y = y.squeeze(1)
    # print("pad", y.size())
    # torch.Size([1, 8192])
    
    # print("n_fft: ", n_fft, " hop_size: ", hop_size, " win_size: ", win_size, " center: ", center)
    # n_fft:  1024  hop_size:  256  win_size:  1024  center:  False
    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
    # print("after stft", spec.size())
    # after stft torch.Size([1, 513, 29])
    
    spec = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + (1e-9))
    # print("after sqrt", spec.size())
    # print("mel_basis", mel_basis[str(fmax)+'_'+str(y.device)].size())

    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    # print("after matmul", spec.size())
    # after matmul torch.Size([1, 80, 29])
    spec = spectral_normalize_torch(spec)
    # print("-------------after spectral_normalize_torch", spec.size())
    # -------------after spectral_normalize_torch torch.Size([1, 80, 29])
    return spec

class MelDataset(torch.utils.data.Dataset):
    def __init__(self, training_files, segment_size, n_fft, num_mels,
                 hop_size, win_size, sampling_rate, fmin, fmax, shuffle=True, n_cache_reuse=1,
                 device=None, fmax_loss=None, fine_tuning=False, base_mels_path=None, split=True):
        self.audio_files = training_files
        random.seed(1234)
        if shuffle:
            random.shuffle(self.audio_files)
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.fmax_loss = fmax_loss
        self.cached_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device
        self.fine_tuning = fine_tuning
        self.base_mels_path = base_mels_path
        self.split = split

    def __getitem__(self, index):
        filename = self.audio_files[index]
        if self._cache_ref_count == 0:
            audio, sampling_rate = load_wav_to_torch(filename)
            if sampling_rate != self.sampling_rate:
                raise ValueError("{} SR doesn't match target {} SR".format(
                    sampling_rate, self.sampling_rate))
            audio_norm = audio / MAX_WAV_VALUE
            audio_norm = audio_norm.unsqueeze(0)
            self.cached_wav = audio_norm
            self._cache_ref_count = self.n_cache_reuse
        else:
            audio_norm = self.cached_wav
            self._cache_ref_count -= 1

        if not self.fine_tuning:
            if self.split:
                if audio_norm.size(1) >= self.segment_size:
                    max_audio_start = audio_norm.size(1) - self.segment_size
                    audio_start = random.randint(0, max_audio_start)
                    audio_norm = audio_norm[:, audio_start:audio_start+self.segment_size]
                else:
                    audio_norm = torch.nn.functional.pad(audio_norm, (0, self.segment_size - audio_norm.size(1)), 'constant')
            # print("norm ---------------",audio_norm.size())
            # norm --------------- torch.Size([1, 8192])
            mel = mel_spectrogram(audio_norm, self.n_fft, self.num_mels,
                                  self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax,
                                  center=False)
        else:
            mel = np.load(
                os.path.join(self.base_mels_path, os.path.splitext(os.path.split(filename)[-1])[0] + '.npy'))
            mel = torch.from_numpy(mel)

            if len(mel.shape) < 3:
                mel = mel.unsqueeze(0)

            if self.split:
                frames_per_seg = math.ceil(self.segment_size / self.hop_size)

                if audio_norm.size(1) >= self.segment_size:
                    mel_start = random.randint(0, mel.size(2) - frames_per_seg - 1)
                    mel = mel[:, :, mel_start:mel_start + frames_per_seg]
                    audio_norm = audio_norm[:, mel_start * self.hop_size:(mel_start + frames_per_seg) * self.hop_size]
                else:
                    mel = torch.nn.functional.pad(mel, (0, frames_per_seg - mel.size(2)), 'constant')
                    audio_norm = torch.nn.functional.pad(audio_norm, (0, self.segment_size - audio_norm.size(1)), 'constant')

        mel_loss = mel_spectrogram(audio_norm.squeeze(0), self.n_fft, self.num_mels,
                                   self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax_loss,
                                   center=False)
        # print(mel.squeeze().size(), audio_norm.squeeze(0).size(), filename, mel_loss.squeeze().size())
        # torch.Size([80, 29]) torch.Size([8192]) ../dataset/VS/SPK087/SPK087YTNSO976/SPK087YTNSO976M003.wav torch.Size([80, 32])
        return (mel.squeeze(), audio_norm.squeeze(0), filename, mel_loss.squeeze())

    def __len__(self):
        return len(self.audio_files)

