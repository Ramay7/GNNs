import pysparnn.cluster_index as ci

import numpy as np
from scipy.sparse import csr_matrix

features = np.random.binomial(1, 0.01, size=(1000, 20000))
features = csr_matrix(features)

# build the search index!
data_to_return = range(1000)
cp = ci.MultiClusterIndex(features, data_to_return)

t = cp.search(features[:5], k=1, return_distance=False)
print(t)