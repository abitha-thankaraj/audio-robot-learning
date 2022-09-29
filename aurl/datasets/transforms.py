import torchaudio

def to_mel_spectrogram(cfg, waveform):
    to_mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=cfg.sample_rate, n_fft=cfg.n_fft, n_mels=cfg.n_mels,
        hop_length=cfg.hop_length, normalized= cfg.normalize)
    return to_mel_spec(waveform)
    
def get_signal(cfg, waveform):
    return waveform
