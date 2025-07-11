import torch
import whisper
import ssl
import librosa
import numpy as np

# Disable SSL verification to avoid certificate issues
ssl._create_default_https_context = ssl._create_unverified_context

def load_audio_with_librosa(file_path, target_sr=16000):
    """Load audio using librosa instead of ffmpeg"""
    # Load audio file with librosa
    audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
    return audio

#
#
#
#
model = whisper.load_model('tiny')
audio = load_audio_with_librosa('name.m4a')
audio = whisper.pad_or_trim(audio)
lg_m1 = whisper.log_mel_spectrogram(audio)
tknsr = whisper.tokenizer.get_tokenizer(multilingual=True)

#
#
#
opt = whisper.DecodingOptions()
res = whisper.decode(model, lg_m1.to(model.device), opt)
print('Baseline:', res.text) # Hello my name is Yurin.
print('------')

#
#
#
ids = []
ids += [tknsr.sot]
ids += [tknsr.language_token]
ids += [tknsr.transcribe]
ids += [tknsr.no_timestamps]
ids += tknsr.encode(' Hello, my name is Ewan.')
ids += [tknsr.eot]

#
#
#
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
criterion = torch.nn.CrossEntropyLoss()

#
#
#
model.train()
tks = torch.tensor(ids).unsqueeze(0).to(model.device)
mel = whisper.log_mel_spectrogram(audio).unsqueeze(0).to(model.device)

#
#
#
pred = model(tokens=tks, mel=mel)
trgt = tks[:, 1:].contiguous()
pred = pred[:, :-1, :].contiguous()

#
#
#
print('Ids Target:', trgt.squeeze().tolist())
print('Ids Output:', torch.argmax(pred, dim=-1).squeeze().tolist())
print('Txt Target:', tknsr.decode(trgt.squeeze().tolist()))
print('Txt Output:', tknsr.decode(torch.argmax(pred, dim=-1).squeeze().tolist()))

#
#
#
loss = criterion(pred.transpose(1, 2), trgt)
print('Loss:', loss.item())
print('------')
optimizer.zero_grad()
loss.backward()
optimizer.step()

#
#
#
model.eval()
prd = model(tokens=tks, mel=mel)
prd = prd[:, :-1, :].contiguous()

#
#
#
print('Ids Target:', trgt.squeeze().tolist())
print('Ids Output:', torch.argmax(prd, dim=-1).squeeze().tolist())
print('Txt Target:', tknsr.decode(trgt.squeeze().tolist()))
print('Txt Output:', tknsr.decode(torch.argmax(prd, dim=-1).squeeze().tolist()))
loss = criterion(prd.transpose(1, 2), trgt)
print('Loss:', loss.item())
