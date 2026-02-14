import torch
import matplotlib.pyplot as plt
import numpy as np

# Load the JS divergence data
js_divs = torch.load('js_divergence.pt')
js_divs = js_divs.cpu().float().numpy()  # [32, 576]

print(f"JS divergence shape: {js_divs.shape}")
print(f"Layers: {js_divs.shape[0]}, Tokens: {js_divs.shape[1]}")

# Transpose: [576, 32] so y-axis is patch token, x-axis is layer
js_divs_T = js_divs.T

fig, ax = plt.subplots(figsize=(12, 16))

# Heatmap with Purples colormap (darker = closer to 1)
im = ax.imshow(
    js_divs_T,
    aspect='auto',
    cmap='Purples',
    vmin=0,
    vmax=1,
    interpolation='nearest'
)

ax.set_xlabel('Layer Transition (i → i+1)', fontsize=12)
ax.set_ylabel('Patch Token Index', fontsize=12)
ax.set_title('JS Divergence Heatmap', fontsize=14)

# Set x-ticks for each layer transition
ax.set_xticks(np.arange(js_divs_T.shape[1]))
ax.set_xticklabels([f'{i}→{i+1}' for i in range(js_divs_T.shape[1])], rotation=45, ha='right', fontsize=8)

# Colorbar
cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('JS Divergence', fontsize=12)

plt.tight_layout()
plt.savefig('js_divergence_visualization.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nStatistics:")
print(f"  Mean JS Divergence: {js_divs.mean():.6f}")
print(f"  Std JS Divergence: {js_divs.std():.6f}")
print(f"  Min JS Divergence: {js_divs.min():.6f}")
print(f"  Max JS Divergence: {js_divs.max():.6f}")
