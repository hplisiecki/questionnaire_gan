import numpy as np
rng = np.random.default_rng()
import pandas as pd

# Random number of items
n=rng.poisson(lam=8)+1

# Random means
means = 2*np.random.random(size=n)-1

# Random covariance matrix
b = 2*np.random.random(size=(n,n))-1
b_symm = (b + b.T)/2

# Matrix of responses
responses = rng.multivariate_normal(means, b_symm, check_valid="ignore", size=300)
responses = pd.DataFrame(responses)

# Empirical values
correlations = responses.corr()
means = responses.mean()

# Order
responses.insert(0, 'Order', range(1, 1+len(responses)))


# Junk
# Random
junk_random = 2*np.random.random(size=(2,n))-1
junk_random = pd.DataFrame(junk)

#Glat




print(junk)
