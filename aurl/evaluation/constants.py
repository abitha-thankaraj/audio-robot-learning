# This file contains the configs used to calculate the similarity score between the audio generated and the test audio.

sample_rate = 44100
resample_rate = 11000

# frame_size, hop_length
amplitude_envelope_cfgs = {
    'fly_swatter' : (64, 32),
    'rattle' : (256, 128),
    'vertical_probing': (256,128),
    'horizontal_probing':  (256, 128),
    'tambourine': (256, 128)
}

num_mics = {
    'fly_swatter' : 2,
    'rattle' : 1,
    'vertical_probing': 2,
    'horizontal_probing': 2,
    'tambourine': 1
}