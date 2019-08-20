

from numpy.random import rand
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

cdict = {'red':   ((0.0, 0.0, 0.0),
                   (0.5, 0.0, 0.0),
                   (1.0, 1.0, 1.0)),
         'blue':  ((0.0, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),
         'green': ((0.0, 0.0, 1.0),
                   (0.5, 0.0, 0.0),
                   (1.0, 0.0, 0.0))}

cmap = mcolors.LinearSegmentedColormap(
'my_colormap', cdict, 100)

ax = plt.subplot(111)
im = ax.imshow(1*rand(20, 20), cmap=cmap)
plt.colorbar(im)
plt.show()