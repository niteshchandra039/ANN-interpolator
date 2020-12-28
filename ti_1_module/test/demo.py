import numpy as np
from sklearn.preprocessing import StandardScaler

from Interpolator import training

print('Reading Files')
TGM = np.loadtxt('/home/nitesh/nitesh/PhD/GSL_Interpolator/TGM.txt')
y = np.loadtxt('/home/nitesh/nitesh/PhD/GSL_Interpolator/spectra.txt')
var = np.loadtxt('/home/nitesh/nitesh/PhD/GSL_Interpolator/variation_spectra.txt')
wl_NGC = np.loadtxt('/home/nitesh/nitesh/PhD/GSL_Interpolator/wl_NGC.txt')

print('Scaling Input Parameters')
scalar = StandardScaler().fit(TGM)
X = scalar.transform(TGM)

# Initialise a model

toy = training.InterpolateModel(tau=1 / 2., n1=32, n2=64, n3=128, out=100, V=var[:, 140:240])  # [:,40:140]))

print('.' * 40 + 'Training' + '.' * 40)
print('.' * 40 + 'Training' + '.' * 40 + '\n')
model, hist, flag = toy.train_model(X, y[:, 40:140], epochs=10, obj='mse', verbosity=True)
# .......save the interpolator file.....
toy.save_model('toy_with_LSS', model)

# .....plot some reconstruction..........
toy.plot(wl=wl_NGC[40:140], X=X, y=y[:, 40:140], TGM=TGM, model=model)
