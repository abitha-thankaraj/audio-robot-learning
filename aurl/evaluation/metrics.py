import torch
import torch.nn.functional as F
import torchaudio.transforms as T
import tslearn.metrics

def resample_audio(audio, sample_rate=44100, resample_rate=11000): # Default audio sampling rates for our dataset
    return T.Resample(sample_rate, resample_rate, dtype=audio.dtype)(audio)

def get_amplitude_envelope(signal, frame_size=256, hop_length=128, num_mics=2): # Shape = n_mics x t
    ae = []
    for j in range(num_mics):
        ae.append(torch.tensor(torch.tensor([max(signal[j][i:i+frame_size]) for i in range(0, len(signal[j]), hop_length)])))
    return torch.stack(ae)


def calculate_raw_similarity_score(audio1, audio2, frame_size, hop_length, num_mics):    
    
    # Downsample audio
    audio1, audio2 = resample_audio(audio1), resample_audio(audio2)

    # Pad both audio sequences to equal length
    max_audio_len = max(audio1.shape[1], audio2.shape[1])
    audio1 = F.pad(audio1, pad=(0, max_audio_len - audio1.shape[1]), mode='constant', value=0.)
    audio2 = F.pad(audio2, pad=(0, max_audio_len - audio2.shape[1]), mode='constant', value=0.)

    # Extract amplitude envelope
    f1, f2 =  get_amplitude_envelope(audio1, frame_size = frame_size, hop_length=hop_length, num_mics=num_mics), \
              get_amplitude_envelope(audio2, frame_size = frame_size, hop_length=hop_length, num_mics=num_mics)

    # Calculate dtw distance
    dist = 0.
    for i in range(num_mics):
        dist += tslearn.metrics.dtw(f1[i], f2[i]) 
    
    return dist/num_mics