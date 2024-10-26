import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

def plot_data(ax, threshold=0.01):
    # Track previously plotted points to avoid redundancy
    plotted_points = []

    # Function to check if a point is covered
    def is_covered(x, y):
        for (px, py) in plotted_points:
            if abs(px - x) < threshold and abs(py - y) < threshold:
                return True
        return False

    # files for demonstration purposes
    files = ['10_1.0_0.025_7_relu.txt']
    files += [f'10_{float(sw)}_0.1_7_tanh.txt' for sw in [1.0, 1.5, 2.0, 4.0, 8.0]]

    colours = {1: '#1f77b4', 2: '#de6610', 4: '#551887', 8: '#d00000', 'relu': 'midnightblue', 1.5: '#ffc90e'}

    total, printed = 0, 0

    for fname in files:
        sigmaw = float(fname.split("_")[1])
        df = pd.read_csv('data/data_1ab/'+fname, sep=' ', names=['count', 'kc'], usecols=[1,2])
        dfsum = np.sum(df['count'])

        plotted_points = []

        # Group the DataFrame by 'kc' to process each unique 'kc' value
        grouped = df.groupby('kc')

        # Iterate over each group
        for kc_value, group in grouped:
            # Extract the 'count' values for this 'kc'
            counts = group['count']
            
            # Find the minimum and maximum 'count' values
            min_count = int(counts.min())
            max_count = int(counts.max())
            
            # Convert the 'count' values to a set for fast lookup
            counts_set = set(counts)
            # further narrow down
            
            # Iterate over each integer between min_count and max_count
            for count in range(min_count, max_count + 1):
                # If the integer 'count' is present in the counts for this 'kc'
                if count in counts_set:
                    # Add the point to the list of plotted points
                    plotted_points.append((kc_value, count))

        # Now 'plotted_points' contains the selected points to plot
        # You can convert it back to a DataFrame if needed
        plot_df = pd.DataFrame(plotted_points, columns=['kc', 'count'])
        plot_df['count'] = plot_df['count'] / dfsum

        print(len(plot_df))

        if 'relu' not in fname:
            ax.scatter(plot_df['kc'], plot_df['count'], label=r'$' + str(sigmaw) + r'\mid 10$', color=colours[sigmaw], s=10, alpha=0.75)
        else:
            ax.scatter(plot_df['kc'], plot_df['count'], label=r'$' + str(sigmaw) + r'\mid 10, relu$', color=colours['relu'], s=10, alpha=0.75)

    # Add the 'random' scatter points separately
    random_x = [108.5 + 3.5 * i for i in range(21)]
    random_y = [1e-8 for _ in range(21)]

    ax.scatter(random_x, random_y, label='random', color='#1d9641', s=10, alpha=0.75)

    # Set up axes labels and formatting
    ax.set_yscale('log')
    ax.set_xlim([0, 160])
    ax.set_ylim([0.8 * 1e-8, 1.2])
    ax.set_ylabel(r'$P(f)$')
    ax.set_xlabel('LZ Complexity, K(f)')
    ax.legend(title=r'$\sigma_w \; \mid \; N_L$', fontsize='x-small', title_fontsize='x-small')
    ax.xaxis.set_major_locator(MultipleLocator(40))
    ax.xaxis.set_minor_locator(MultipleLocator(20))

    locmin = matplotlib.ticker.LogLocator(base=10.0, subs=(0.2, 0.4, 0.6, 0.8, 1.0), numticks=12)
    ax.yaxis.set_minor_locator(locmin)
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    locmax = matplotlib.ticker.LogLocator(base=10.0, numticks=1)
    
    ax.set_yticks([10**(-1*2*i) for i in [4, 3, 2, 1, 0]])

    return ax

# Main script
fig, ax = plt.subplots()
ax.margins(0.05)

ax = plot_data(ax)

fig.set_size_inches(3, 2.4)
plt.tight_layout()
plt.savefig(f'plots/1b.pdf', dpi=300, bbox_inches='tight')
