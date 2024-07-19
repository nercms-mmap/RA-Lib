import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_scatter_with_lines_and_points(file_path, x_column, y_column, method_column, group_column, type_column, start_row, end_row, fig_width, fig_height, colors):
    
    # load_data
    data = pd.read_csv(file_path)

    data_subset = data.iloc[start_row:end_row]

    x = data_subset[x_column]
    y = data_subset[y_column]
    method = data_subset[method_column]
    group = data_subset[group_column]
    dtype = data_subset[type_column]

    #colors
    if colors is None:
        colors = plt.cm.tab10(range(10))

    # markers
    markers = ['o', 's', '^', 'D', 'x', '+', '*', '|', '_', 'P']
    
    # size
    plt.figure(figsize=(fig_width, fig_height))

    # for single

    single_points = data_subset[dtype == 'single']
    np.random.seed(0)
    marker_styles = np.random.choice(markers, size=len(single_points))
    for i in range(len(single_points)):
        plt.scatter(single_points[x_column].iloc[i], single_points[y_column].iloc[i],  marker=marker_styles[i], label=single_points[method_column].iloc[i], alpha=0.7)

    # line (group)
    grouped_points = data_subset[dtype == 'group']
    unique_groups = grouped_points[group_column].dropna().unique()

    if len(colors) < len(unique_groups):
        raise ValueError("lack of colors")

    color_map = dict(zip(unique_groups, colors))  
    for g in unique_groups:
        group_data = grouped_points[grouped_points[group_column] == g]
        plt.plot(group_data[x_column], group_data[y_column], color=color_map[g], label=f'Group {g}')
        plt.scatter(group_data[x_column], group_data[y_column])  

    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.title('Execution time in CUHK03 (labeled)')

    # position of legends
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()

# settings
file_path = 'D:\\Code of RA\\results\\re-ID-CUHK03labeled_ex-time.csv'  
x_column = 's'  # x 
y_column = 'MAP'  # y
method_column = 'Method'
group_column = 'Group'  
type_column = 'Type'  
start_row = 6  
end_row = 30   
fig_width = 12 
fig_height = 8  

custom_colors = ['red', 'blue', 'green', 'orange', 'purple']

plot_scatter_with_lines_and_points(file_path, x_column, y_column, method_column, group_column, type_column, start_row, end_row, fig_width, fig_height, colors=custom_colors)