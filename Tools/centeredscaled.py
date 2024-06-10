def centeredscaled(x):
  
  import numpy as np

  muX = x.mean(0)
  X0 = x - muX

  ssX = (X0**2.).sum()

  # centred Frobenius norm
  normX = np.sqrt(ssX)

  # scale to equal (unit) norm
  X0 = X0 / normX
  
  return X0