import matplotlib.pyplot as plt
import torch
from data import ds

# Get a sample from your dataset
sample_idx = 0  # Change this to any index you want
sample = ds[sample_idx]

# Extract the spectrogram data
spectrogram = sample['audio']  # This should be your processed spectrogram
label = sample['label']

# Convert to numpy if it's a tensor
if torch.is_tensor(spectrogram):
    spec_np = spectrogram.squeeze().numpy()  # Remove channel dim and convert to numpy
else:
    spec_np = spectrogram.squeeze()

# Plot the spectrogram
plt.figure(figsize=(12, 8))
plt.imshow(spec_np, aspect='auto', origin='lower', cmap='viridis')
plt.colorbar(label='Log Magnitude')
plt.title(f'Mel Spectrogram - (Label: {label})')
plt.xlabel('Time Frames')
plt.ylabel('Mel Frequency Bins')
plt.tight_layout()
plt.show()

# If you want to plot multiple samples
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for i in range(6):
    sample = ds[i]
    spectrogram = sample['audio']
    class_id = sample['label']
    
    if torch.is_tensor(spectrogram):
        spec_np = spectrogram.squeeze().numpy()
    else:
        spec_np = spectrogram.squeeze()
    
    row, col = i // 3, i % 3
    axes[row, col].imshow(spec_np, aspect='auto', origin='lower', cmap='viridis')
    axes[row, col].set_title(f'Class: {class_id}')
    axes[row, col].set_xlabel('Time')
    axes[row, col].set_ylabel('Mel Bins')

plt.tight_layout()
plt.show()