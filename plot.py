import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib.lines import Line2D
from matplotlib import cm
from matplotlib.colors import Normalize, LinearSegmentedColormap

def adjust_middle_brightness(cmap, n_colors=256, middle_factor=0.95):
    colors = cmap(np.linspace(0, 1, n_colors)) 
    middle_index = int(n_colors / 2)
    for i in range(middle_index - 30, middle_index + 30): 
        colors[i] = np.array(colors[i]) * middle_factor

    new_cmap = LinearSegmentedColormap.from_list("adjusted_RdYlBu", colors)
    return new_cmap

plt.rcParams["font.family"] = "Times New Roman"  
plt.rcParams["font.size"] = 18  

df = pd.read_excel(r'D:\Code of RA\实验结果\CUSR_experiments_map.xlsx', header=None)

series_names = df.iloc[2:35, 0].values  
datasets = df.iloc[0, 2:6].values  

x_positions = np.arange(4)

markers = ['o', 's', '^', 'D', 'x', 'X', '+', 'v', 'H', '*', '|', '_', 'P']
cmap = plt.cm.RdYlBu
adjusted_cmap = adjust_middle_brightness(cmap, n_colors=256, middle_factor=0.94)
colors = adjusted_cmap(np.linspace(0, 1, len(series_names)))
# colors = plt.cm.get_cmap('RdYlBu', len(series_names))
marker_styles = markers[:len(series_names)]

plt.figure(figsize=(20, 16))
scatter_handles = []
for i, series_name in enumerate(series_names):

    values = df.iloc[i + 2, 2:6].values 

    color = colors[i % len(colors)]
    marker = marker_styles[i % len(marker_styles)] 

    for j in range(4): 

        x_position = np.interp(i, [0, len(series_names)-1], [x_positions[j], x_positions[j] + 0.9])
        x_position = x_position + 0.05
        # plot
        scatter = plt.scatter(x_position, values[j], marker=marker, color=color, s=150) 
    
    scatter_handles.append(scatter)

# plt.title('Comparison of different methods')
# plt.xlabel('Datasets')
plt.ylabel('mAP (%)')

xticks_positions = x_positions + 0.5
plt.xticks(xticks_positions, datasets) 


plt.axvline(x=0, color='gray', linestyle='--', linewidth=1.5)
plt.axvline(x=1, color='gray', linestyle='--', linewidth=1.5)
plt.axvline(x=2, color='gray', linestyle='--', linewidth=1.5)
plt.axvline(x=3, color='gray', linestyle='--', linewidth=1.5)
plt.axvline(x=4, color='gray', linestyle='--', linewidth=1.5)
plt.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)

ax = plt.gca()

ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)

plt.legend(scatter_handles, series_names, title="", bbox_to_anchor=(1.01, 0.95), loc='upper left')

plt.tight_layout()

plt.savefig(r"D:\Code of RA\实验结果\plot_4_re_id-dataset.pdf", format="pdf")

plt.show()