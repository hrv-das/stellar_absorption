import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling import models,fitting
import os
import warnings

os.chdir('/home/hrv/Downloads')
warnings.filterwarnings('ignore')
data = np.loadtxt('absorption.dat')
x = data[:,0]
y = data[:,1]

my_model1 = models.Gaussian1D(amplitude = -1, mean = 656, stddev = 1) + models.Const1D(amplitude = 1)
fit_method = fitting.LevMarLSQFitter()
my_bestfit1 = fit_method(my_model1,x,y,)

plt.figure()
plt.plot(x,y,'c+',label='Data')
plt.plot(x,my_bestfit1(x),'r-',label='Fitted Curve',alpha=0.8)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Flux (AU)')
plt.grid(True,which='both')
plt.legend(loc='best')
plt.show()

my_model2 = models.Voigt1D(x_0 = 656, amplitude_L = -1, fwhm_L = 1, fwhm_G = 1) + models.Const1D(amplitude = 1)
fit_method = fitting.LevMarLSQFitter()
my_bestfit2 = fit_method(my_model2,x,y,)

plt.figure()
plt.plot(x,y,'c+',label='Data')
plt.plot(x,my_bestfit2(x),'r-',label='Fitted Curve',alpha=0.8)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Flux (AU)')
plt.grid(True,which='both')
plt.legend(loc='best')
plt.show()

my_model3 = models.Lorentz1D(amplitude = -1, x_0 = 656, fwhm = 1) + models.Const1D(amplitude = 1)
fit_method = fitting.LevMarLSQFitter()
my_bestfit3 = fit_method(my_model3,x,y,)

plt.figure()
plt.plot(x,y,'c+',label='Data')
plt.plot(x,my_bestfit3(x),'r-',label='Fitted Curve',alpha=0.8)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Flux (AU)')
plt.grid(True,which='both')
plt.legend(loc='best')
plt.show()
