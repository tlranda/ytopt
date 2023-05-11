import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patches
import seaborn as sns
import numpy as np
import pandas as pd

# Ensure consistent data
np.random.seed(1234)
from gc_vis import source_data, model_covariance

fig = plt.figure()

# First subplot: x0
axis_x0 = fig.add_subplot(221)
sns.histplot(source_data['x0'], ax=axis_x0, legend=False, stat='proportion')

# Second subplot: y
axis_y = fig.add_subplot(222)
sns.histplot(source_data['y'], ax=axis_y, legend=False, stat='proportion')
axis_y.yaxis.tick_right()
axis_y.yaxis.set_label_position('right')

# Third subplot: size
axis_size = fig.add_subplot(223)
sns.histplot(source_data['size'], ax=axis_size, legend=False, stat='proportion')

# Fourth submplot: covariance
axis_covariance = fig.add_subplot(224)
axis_covariance.plot([0,1],[0,1])
axis_covariance.set_xlabel('Correlation Influence')
axis_covariance.yaxis.tick_right()
#mask = np.triu(np.ones_like(model_covariance, dtype=bool))
#cmap = sns.diverging_palette(230, 20, as_cmap=True)
# Rename index
#model_covariance = model_covariance.rename(columns={'x0.value': 'x0','y.value': 'y', 'size#1#3.value': 'size'}, index={'x0.value': 'x0', 'y.value': 'y', 'size#1#3.value': 'size'})
#sns.heatmap(model_covariance, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=axis_covariance)

# Add lines between subplots
xyCovariance = [0.75, 0.75]
xyX0 = [75, 0.3]
axis_x0.plot(*xyX0, "o")
xyY = [23,0.25]
axis_y.plot(*xyY, "o")
xySize = [1.9, 0.5]
axis_size.plot(*xySize, "o")
# ConnectionPatch handles the transform internally so no need to get fig.transFigure
x0_arrow = patches.ConnectionPatch(
    xyCovariance,
    xyX0,
    coordsA=axis_covariance.transData,
    coordsB=axis_x0.transData,
    # Default shrink parameter is 0 so can be omitted
    color="black",
    arrowstyle="-|>",  # "normal" arrow
    mutation_scale=15,  # controls arrow head size
    linewidth=1,
    zorder=1,
)
y_arrow = patches.ConnectionPatch(
    xyCovariance,
    xyY,
    coordsA=axis_covariance.transData,
    coordsB=axis_y.transData,
    color="black",
    arrowstyle="-|>",
    mutation_scale=15,
    linewidth=1,
    zorder=1,
)
size_arrow = patches.ConnectionPatch(
    xyCovariance,
    xySize,
    coordsA=axis_covariance.transData,
    coordsB=axis_size.transData,
    color="black",
    arrowstyle="-|>",
    mutation_scale=15,
    linewidth=1,
    zorder=1,
)
fig.patches.extend([x0_arrow, y_arrow, size_arrow])
axis_covariance.plot(*xyCovariance, "o", zorder=2)
fig.tight_layout()

# Show figure
fig.savefig("Assets/UnconditionalExample.png", format="png")

