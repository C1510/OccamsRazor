import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from matplotlib.ticker import MultipleLocator

def plot_data(ax):
    # Files for demonstration purposes
    files = ['10_1.0_0.025_7_relu.txt']
    files += [f'10_{float(sw)}_0.1_7_tanh.txt' for sw in [1.0, 1.5, 2.0, 4.0, 8.0]]

    colours = {1: '#1f77b4', 2: '#de6610', 4: '#551887', 8: '#d00000', 'relu': 'midnightblue', 1.5: '#ffc90e'}

    # List to hold all points along with their plotting order and other info
    all_points = {}
    max_points = {}

    plotting_index = 0  # To keep track of the plotting order
    # For each file, collect the points
    for fname in files:
        print('doing')
        sigmaw = float(fname.split("_")[1]) if 'relu' not in fname else 0
        df = pd.read_csv('data/data_1ab/'+fname, sep=' ', names=['count', 'kc'], usecols=[1,2])
        dfsum = np.sum(df['count'])
        all_points[sigmaw]={}

        # Group the DataFrame by 'kc' to process each unique 'kc' value
        grouped = df.groupby('kc')
        max_points[sigmaw] = {}
        # Iterate over each group
        for kc_value, group in grouped:
            # Extract the 'count' values for this 'kc'
            counts = group['count']

            # Find the minimum and maximum 'count' values
            min_count = int(counts.min())
            max_count = int(counts.max())
            max_points[sigmaw][kc_value] = max_count / dfsum

            # Convert the 'count' values to a set for fast lookup
            counts_set = set(counts)

            # Iterate over each integer between min_count and max_count
            for count in range(min_count, max_count + 1):
                # If the integer 'count' is present in the counts for this 'kc'
                if count in counts_set:
                    # Normalize the 'count' value
                    y_value = count / dfsum
                    x_value = kc_value

                    # Store the point along with its plotting info
                    if x_value in all_points[sigmaw]:
                        all_points[sigmaw][x_value].add(y_value)
                    else:
                        all_points[sigmaw][x_value] = set([y_value])

    # import pickle
    # with open('all_points.pkl','wb') as pklfile:
    #     pickle.dump(all_points, pklfile)
    # with open('max_points.pkl','wb') as pklfile:
    #     pickle.dump(max_points, pklfile)
    # with open('all_points.pkl','rb') as pklfile:
    #     all_points = pickle.load(pklfile)
    # with open('max_points.pkl','rb') as pklfile:
    #     max_points = pickle.load(pklfile)

    points_plotted = 0
    to_plot = {sigmaw:[] for sigmaw in all_points}
    for sigmaw, data in all_points.items():
        for x, values in data.items():
            try:
                maxs = max([max_points[sw].get(x, 0) for sw in max_points if sw>sigmaw])
            except:
                maxs = 0
            for y in values:
                if x < 25:
                    to_plot[sigmaw].append((x, y))
                    points_plotted+=1
                elif y > 1e-5:
                    to_plot[sigmaw].append((x, y))
                    points_plotted+=1
                elif y>maxs:
                    to_plot[sigmaw].append((x, y))
                    points_plotted+=1

    print(points_plotted)

    for sigmaw, data in to_plot.items():
        xs = [i[0] for i in data]
        ys = [i[1] for i in data]
        if sigmaw != 0:
            ax.scatter(xs, ys, label=r'$' + str(sigmaw) + r'\mid 10$', color=colours[sigmaw], s=10, alpha=0.75)
        else:
            ax.scatter(xs, ys, label=r'$' + str(1.0) + r'\mid 10, relu$', color=colours['relu'], s=10, alpha=0.75)


    # Add the 'random' scatter points separately
    random_x = [108.5 + 3.5 * i for i in range(21)]
    random_y = [1e-8 for _ in range(21)]

    ax.scatter(random_x, random_y, label='random', color='#1d9641', s=10, alpha=0.75)

    # Set up axes labels and formatting
    ax.set_yscale('log')
    ax.set_xlim([0, 160])
    ax.set_ylim([0.8 * 1e-8, 1.2])
    ax.set_ylabel(r'$P(f)$')
    ax.set_xlabel('LZ Complexity, $K(f)$')
    ax.legend(title=r'$\sigma_w \; \mid \; N_L$', fontsize='x-small', title_fontsize='x-small')
    ax.xaxis.set_major_locator(MultipleLocator(40))
    ax.xaxis.set_minor_locator(MultipleLocator(20))

    locmin = matplotlib.ticker.LogLocator(base=10.0, subs=(0.2, 0.4, 0.6, 0.8, 1.0), numticks=12)
    ax.yaxis.set_minor_locator(locmin)
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_yticks([10**(-2*i) for i in [4, 3, 2, 1, 0]])

    return ax

# Main script
fig, ax = plt.subplots()
ax.margins(0.05)

ax = plot_data(ax)

fig.set_size_inches(3, 2.4)
plt.tight_layout()
plt.savefig('plots/1b.pdf', dpi=300, bbox_inches='tight')
