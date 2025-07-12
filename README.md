Welcome to the linefit.py project

The script includes the SpectralLineFitter() class, to be imported into the user's personal projects. 
The class is built upon lmfit and can fit absorption, emission or P-Cygni line profiles via SpectralLineFitter().fit_absorption, .fit_emission or .fit_pcygni. 
It is important to mention that the data must be previously normalised.
The fit returns a dictionary containing:

\begin{enumerate}
\setcounter{enumi}{-1}
    \item \texttt{result}. The resulting model from \texttt{lmfit}.
    \item \texttt{x\_fit}. The wavelength values used for the fit (from data).
    \item \texttt{y\_fit}. The flux values used for the fit (from data).
    \item \texttt{model\_y}. The fitted flux values.
    \item \texttt{model\_y\_error}. Standard deviation of the last entry.
    \item \texttt{absorption\_center}. The wavelength at the minimum flux value.
    \item \texttt{absorption\_center\_error}. Uncertainty of the last entry.
    \item \texttt{fwhm\_absorption}. Estimated full width at half maximum of the fit.
    \item \texttt{fwhm\_absorption\_error}. Uncertainty of the last entry.
    \texttt{ew\_absorption}. Estimated equivalent width of the fit.
    \item \texttt{ew\_absorption\_error}. Uncertainty of the last entry.
    \item \texttt{rv\_absorption}. Estimated radial velocity using Doppler's formula, \texttt{center\_line} and \texttt{absorption\_center}.
    \item \texttt{rv\_absorption\_error}. Uncertainty of the last entry.
\end{enumerate}

For the EW, the script includes a _monte_carlo_ew_error method to estimate the errors using Monte Carlo. The error in RV is calculated using error propagation.
The script is parallelised ThreadPoolExecutor.

Quick example:

import pandas as pd 
from linefit import SpectralLineFitter
import matplotlib.pyplot as plt
fitter = SpectralLineFitter() # Initialize the class
data = pd.read_csv("HD75149.txt", sep="\t", header=None, names=["Wavelenght", "Flux"]) # Read my data.
fit = fitter.fit_absorption(data["Wavelenght"].values, data["Flux"].values, center_line=4387.93, fit_width=10, n_mc=100, n_workers=5) # The class fits the line 4387.93 using a 10 Angstrom window 
                                                                                                                                      # (from 4387.93 - 10/2 to 4387.93 + 10/2) and compute errors 
                                                                                                                                      # using Monte Carlo with 100 workers in this case. 
                                                                                                                                      # Computations are parallelised using 5 cores.
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.set_xlabel(r"Wavelenght [$\AA$]")
ax.set_ylabel("Normalized flux")
ax.scatter(fit["x_fit"], fit["y_fit"], c="r", label="Data")
ax.plot(fit["x_fit"], fit["model_y"], c="k", label="Fit")
