import numpy as np
from ripser import ripser
from persim import plot_diagrams

data = np.random.random((100,512))
diagrams = ripser(data, maxdim=5)['dgms']
plot_diagrams(diagrams, show=True)