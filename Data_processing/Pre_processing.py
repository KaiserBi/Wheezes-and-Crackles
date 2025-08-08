import os
import torchaudio
import torch

#extracts segments from the .wav files and pairing them with labels within annotation txts
def segment_extraction (wav_path, label_path, save_dir, sample_rate=16000):

    label_path = os.path.normpath(label_path)
    os.makedirs(save_dir,exist_ok=True)

    waveform, sr = torchaudio.load(wav_path)
    waveform = waveform.mean(0)

    if(sr != sample_rate):
        resamp = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
        waveform = resamp(waveform)
    
    with open(label_path, mode='r') as f:
        lines = f.readlines()

        for i, line in enumerate(lines):
            parts = line.strip().split()
            
            start, end, crackles, wheezes = float(parts[0]), float(parts[1]), int(parts[2]), int(parts[3])

            start_sampl = int(start*sample_rate)
            end_sampl = int(end*sample_rate)
            segment = waveform[start_sampl:end_sampl]

            data = {'waveform':segment, 'label': torch.tensor([crackles,wheezes]),
                    'start_time': start, 'end_time': end, 'file': os.path.basename(wav_path)}
            
            output_name = f"{os.path.splitext(os.path.basename(wav_path))[0]}_{i}.pt"

            torch.save(data, os.path.join(save_dir, output_name)) 





    
    

    





