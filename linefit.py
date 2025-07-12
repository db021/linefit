import numpy as np
from lmfit.models import ConstantModel, VoigtModel
from concurrent.futures import ThreadPoolExecutor
import copy
from scipy.constants import c  # c in meters per second
c_km_s = c / 1000 # c ≈ 299792.458 km/s

class SpectralLineFitter:
    def __init__(self):
        pass

    @staticmethod
    def _guess_initials(x, y):
        """
        Estimate initial parameters for spectral line fitting.

        Parameters:
            x (np.ndarray): Wavelength array.
            y (np.ndarray): Flux array.

        Returns:
            dict: Estimated parameters including continuum, absorption and emission amplitudes and centers.
        """
        continuum = np.median(y)
        min_idx = np.argmin(y)
        max_idx = np.argmax(y)
        return {
            'continuum': continuum,
            'abs_amp': continuum - y[min_idx],
            'abs_center': x[min_idx],
            'em_amp': y[max_idx] - continuum,
            'em_center': x[max_idx]
        }

    @staticmethod
    def _fit_model(x, y, model, params):
        """
        Fit the model to the data using lmfit.

        Parameters:
            x (np.ndarray): Wavelength array.
            y (np.ndarray): Flux array.
            model (lmfit.Model): Model to be fitted.
            params (lmfit.Parameters): Initial parameters for fitting.

        Returns:
            lmfit.model.ModelResult: The fit result.
        """
        return model.fit(y, params, x=x)

    @staticmethod
    def _voigt_fwhm(sigma, gamma):
        """
        Compute the Full Width at Half Maximum (FWHM) of a Voigt profile.

        Parameters:
            sigma (float): Gaussian sigma of the Voigt profile.
            gamma (float): Lorentzian gamma of the Voigt profile.

        Returns:
            float: Estimated FWHM of the Voigt profile.
        """
        return 0.5346 * 2 * gamma + np.sqrt(0.2166 * (2 * gamma)**2 + (2.355 * sigma)**2)

    @staticmethod
    def _compute_ew(x, y, continuum):
        """
        Compute the equivalent width (EW) of a spectral line.

        Parameters:
            x (np.ndarray): Wavelength array.
            y (np.ndarray): Flux array including the line.
            continuum (float): Continuum level.

        Returns:
            float: Equivalent width (EW) of the line.
        """
        return np.trapz(1 - y / continuum, x)

    @staticmethod
    def _monte_carlo_ew_error(x, y_fit, model, params, component_name, n_mc=100, n_workers=4):
        """
        Estimate the uncertainty on the equivalent width using Monte Carlo sampling.

        Parameters:
            x (np.ndarray): Wavelength array.
            y_fit (np.ndarray): Fitted flux array.
            model (lmfit.Model): Model used for fitting.
            params (lmfit.Parameters): Fit parameters.
            component_name (str): Prefix of the line component.
            n_mc (int): Number of Monte Carlo iterations.
            n_workers (int): Number of threads for parallel processing.

        Returns:
            float: Standard deviation of EW from Monte Carlo simulations.
        """
        residuals = y_fit - model.eval(params=params, x=x)
        std_resid = np.std(residuals)

        def single_sim():
            y_perturbed = y_fit + np.random.normal(0, std_resid, size=y_fit.size)
            pert_result = model.fit(y_perturbed, copy.deepcopy(params), x=x)
            pert_comps = pert_result.eval_components(x=x)
            continuum = pert_result.params['const_c'].value
            ew_pert = SpectralLineFitter._compute_ew(x, pert_comps[component_name] + continuum, continuum)
            return ew_pert

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            ew_samples = list(executor.map(lambda _: single_sim(), range(n_mc)))

        return np.std(ew_samples)

    @staticmethod
    def compute_radial_velocity(observed_center, center_error, rest_wavelength):
        """
        Compute the radial velocity (RV) and its uncertainty using the classical Doppler formula.

        Parameters
        ----------
        observed_center : float
            The observed center wavelength of the line in the same units as rest_wavelength.
        center_error : float
            The uncertainty on the observed center wavelength.
        rest_wavelength : float
            The known rest-frame wavelength of the spectral line.

        Returns
        -------
        rv : float
            Radial velocity in km/s. Positive means redshifted (receding).
        rv_error : float
            Uncertainty in the radial velocity in km/s.
        """
        delta_lambda = observed_center - rest_wavelength
        rv = (delta_lambda / rest_wavelength) * c_km_s
        rv_error = (center_error / rest_wavelength) * c_km_s
        return rv, rv_error


    def fit_emission(self, x, y, center_line, fit_width, n_mc=100, n_workers=4):
        """
        Fit a spectral emission line using a Voigt profile plus a constant continuum.

        Parameters:
        -----------
        x (np.ndarray): Array of wavelength or velocity values (must be same shape as y).
        y (np.ndarray): Array of flux values corresponding to x.
        center_line (float): Expected central wavelength of the emission line.
        fit_width (float): Width of the wavelength window centered at center_line to select data for fitting.
        n_mc (int, optional): Number of Monte Carlo simulations to estimate equivalent width (EW) uncertainty. Default is 100.
        n_workers (int, optional): Number of parallel threads used during Monte Carlo simulations. Default is 4.

        Returns:
        --------
        dict
            A dictionary with the following keys:
            - 'result': lmfit.ModelResult object containing fit details.
            - 'x_fit': The x values used for fitting (subset of x).
            - 'y_fit': The y values used for fitting (subset of y).
            - 'model_y': The best-fit model evaluated on x_fit.
            - 'model_y_err': Standard deviation of residuals between model and data.
            - 'emission_center': Best-fit central position of the emission Voigt profile.
            - 'emission_center_error': 1-sigma uncertainty of the emission center.
            - 'fwhm_emission': Full width at half maximum (FWHM) of the Voigt emission line.
            - 'fwhm_emission_error': Uncertainty in the FWHM derived from error propagation.
            - 'ew_emission': Equivalent width of the emission line (negative for emission).
            - 'ew_emission_error': Uncertainty in the EW from Monte Carlo simulations.
            - 'rv_emission': Radial velocity shift derived from the emission line center in km/s.
            - 'rv_emission_error': Uncertainty in the radial velocity.
        """
        mask = (x >= center_line - fit_width/2) & (x <= center_line + fit_width/2)
        x_fit = x[mask]
        y_fit = y[mask]

        guesses = self._guess_initials(x_fit, y_fit)

        const_mod = ConstantModel(prefix='const_')
        em_mod = VoigtModel(prefix='em_')
        model = const_mod + em_mod
        params = model.make_params()

        params['const_c'].set(value=guesses['continuum'])
        params['em_amplitude'].set(value=guesses['em_amp'], min=0)
        params['em_center'].set(value=guesses['em_center'], min=center_line - fit_width/2, max=center_line + fit_width/2)
        params['em_sigma'].set(value=fit_width / 15, min=1e-4)
        params['em_gamma'].set(value=fit_width / 20, min=1e-4)

        result = self._fit_model(x_fit, y_fit, model, params)
        comps = result.eval_components(x=x_fit)
        continuum = result.params['const_c'].value

        center = result.params['em_center'].value
        sigma = result.params['em_sigma'].value
        gamma = result.params['em_gamma'].value
        center_err = result.params['em_center'].stderr or 0.0
        sigma_err = result.params['em_sigma'].stderr or 0.0
        gamma_err = result.params['em_gamma'].stderr or 0.0

        fwhm = self._voigt_fwhm(sigma, gamma)
        dfwhm_dsigma = (2.355**2 * sigma) / np.sqrt(0.2166 * (2*gamma)**2 + (2.355 * sigma)**2)
        dfwhm_dgamma = (0.5346 * 2 + (0.2166 * 4 * gamma) / (2 * np.sqrt(0.2166 * (2 * gamma)**2 + (2.355 * sigma)**2)))
        fwhm_err = np.sqrt((dfwhm_dsigma * sigma_err)**2 + (dfwhm_dgamma * gamma_err)**2)

        model_y = result.best_fit
        model_y_err = np.std(y_fit - model_y)
        ew = self._compute_ew(x_fit, comps['em_'] + continuum, continuum)
        ew_err = self._monte_carlo_ew_error(x_fit, y_fit, model, params, 'em_', n_mc=n_mc, n_workers=n_workers)
        rv, rv_err = self.compute_radial_velocity(center, center_err, rest_wavelength=center_line)

        return {
            'result': result,
            'x_fit': x_fit,
            'y_fit': y_fit,
            'model_y': model_y,
            'model_y_err': model_y_err,
            'emission_center': center,
            'emission_center_error': center_err,
            'fwhm_emission': fwhm,
            'fwhm_emission_error': fwhm_err,
            'ew_emission': ew,
            'ew_emission_error': ew_err,
            'rv_emission': rv,
            'rv_emission_error': rv_err
        }

    def fit_absorption(self, x, y, center_line, fit_width, n_mc=100, n_workers=4):
        """
        Fit an absorption spectral line profile modeled as a Voigt profile plus a constant continuum.
    
        Parameters:
            x (np.ndarray): Wavelength array.
            y (np.ndarray): Flux array corresponding to wavelengths.
            center_line (float): Rest-frame central wavelength of the line.
            fit_width (float): Width of the wavelength window centered at center_line to select data for fitting.
            n_mc (int, optional): Number of Monte Carlo simulations to estimate equivalent width (EW) uncertainty (default=100).
            n_workers (int, optional): Number of parallel workers for Monte Carlo simulations (default=4).
    
        Returns:
            dict: A dictionary containing detailed fit results with the following keys:
                - 'result' (lmfit.ModelResult): The full fit result object with parameter estimates and statistics.
                - 'x_fit' (np.ndarray): The subset of wavelengths used for fitting.
                - 'y_fit' (np.ndarray): The subset of fluxes used for fitting.
                - 'model_y' (np.ndarray): The best-fit model flux values at x_fit.
                - 'model_y_err' (float): Standard deviation of the residuals (y_fit - model_y), indicating fit quality.
                - 'absorption_center' (float): Best-fit center wavelength of the absorption line.
                - 'absorption_center_error' (float): Uncertainty (standard error) of the absorption center.
                - 'fwhm_absorption' (float): Full Width at Half Maximum (FWHM) of the absorption Voigt profile.
                - 'fwhm_absorption_error' (float): Uncertainty of the absorption FWHM.
                - 'ew_absorption' (float): Equivalent Width (EW) of the absorption line calculated from the model.
                - 'ew_absorption_error' (float): Uncertainty of the EW estimated via Monte Carlo perturbations.
                - 'rv_absorption' (float): Radial velocity shift derived from the absorption line center in km/s.
                - 'rv_absorption_error' (float): Uncertainty in the radial velocity.
    
        Method:
            - Selects a wavelength window around center_line with width fit_width.
            - Estimates initial parameters for the continuum and absorption profile from the data.
            - Builds a model consisting of a constant continuum plus a single Voigt profile with negative amplitude (absorption).
            - Fits the model to the data using lmfit's nonlinear least squares fitting.
            - Extracts fit parameters and calculates the Voigt profile FWHM with error propagation.
            - Computes the equivalent width from the fitted model.
            - Estimates uncertainties on EW by performing parallelized Monte Carlo simulations,
              perturbing the data with residual noise and refitting to build EW distributions.
            - Computes the radial velocity from the classical doppler formula.
            - Returns the full fit result, fit arrays, model fluxes, and all relevant line parameters with uncertainties.
    
        Notes:
            - The absorption amplitude is constrained to be <= 0 (negative) to ensure physical absorption.
            - The Monte Carlo error estimation is parallelized using ThreadPoolExecutor for efficiency.
            - The Voigt FWHM calculation follows the standard empirical formula combining Gaussian sigma and Lorentzian gamma.
        """

        mask = (x >= center_line - fit_width/2) & (x <= center_line + fit_width/2)
        x_fit = x[mask]
        y_fit = y[mask]

        guesses = self._guess_initials(x_fit, y_fit)

        const_mod = ConstantModel(prefix='const_')
        abs_mod = VoigtModel(prefix='abs_')
        model = const_mod + abs_mod
        params = model.make_params()

        params['const_c'].set(value=guesses['continuum'])
        params['abs_amplitude'].set(value=-guesses['abs_amp'], max=0)
        params['abs_center'].set(value=guesses['abs_center'], min=center_line - fit_width/2, max=center_line + fit_width/2)
        params['abs_sigma'].set(value=fit_width / 15, min=1e-4)
        params['abs_gamma'].set(value=fit_width / 20, min=1e-4)

        result = self._fit_model(x_fit, y_fit, model, params)
        comps = result.eval_components(x=x_fit)
        continuum = result.params['const_c'].value

        center = result.params['abs_center'].value
        sigma = result.params['abs_sigma'].value
        gamma = result.params['abs_gamma'].value
        center_err = result.params['abs_center'].stderr or 0.0
        sigma_err = result.params['abs_sigma'].stderr or 0.0
        gamma_err = result.params['abs_gamma'].stderr or 0.0

        fwhm = self._voigt_fwhm(sigma, gamma)
        dfwhm_dsigma = (2.355**2 * sigma) / np.sqrt(0.2166 * (2*gamma)**2 + (2.355 * sigma)**2)
        dfwhm_dgamma = (0.5346 * 2 + (0.2166 * 4 * gamma) / (2 * np.sqrt(0.2166 * (2 * gamma)**2 + (2.355 * sigma)**2)))
        fwhm_err = np.sqrt((dfwhm_dsigma * sigma_err)**2 + (dfwhm_dgamma * gamma_err)**2)

        model_y = result.best_fit
        model_y_err = np.std(y_fit - model_y)
        ew = self._compute_ew(x_fit, comps['abs_'] + continuum, continuum)
        ew_err = self._monte_carlo_ew_error(x_fit, y_fit, model, params, 'abs_', n_mc=n_mc, n_workers=n_workers)
        rv, rv_err = self.compute_radial_velocity(center, center_err, rest_wavelength=center_line)


        return {
            'result': result,
            'x_fit': x_fit,
            'y_fit': y_fit,
            'model_y': model_y,
            'model_y_err': model_y_err,
            'absorption_center': center,
            'absorption_center_error': center_err,
            'fwhm_absorption': fwhm,
            'fwhm_absorption_error': fwhm_err,
            'ew_absorption': ew,
            'ew_absorption_error': ew_err,
            'rv_absorption': rv,
            'rv_absorption_error': rv_err
        }

    def fit_pcygni(self, x, y, center_line, fit_width, n_mc=100, n_workers=4):
        """
        Fit a P Cygni profile consisting of a combined Voigt emission and absorption line plus a constant continuum.

        Parameters:
            x (np.ndarray): Wavelength array.
            y (np.ndarray): Flux array.
            center_line (float): Approximate central wavelength around which to fit.
            fit_width (float): Width of the wavelength fitting window around center_line.
            n_mc (int, optional): Number of Monte Carlo simulations to estimate EW uncertainty (default=100).
            n_workers (int, optional): Number of parallel workers for Monte Carlo simulations (default=4).

        Returns:
            dict: Dictionary with keys:
                - 'result': lmfit.ModelResult object of the combined fit.
                - 'x_fit', 'y_fit': Arrays of the data used for fitting.
                - 'model_y': Best-fit model flux array.
                - 'model_y_err': Standard deviation of residuals (fit quality).
                - 'emission_center': Emission line center wavelength.
                - 'emission_center_error': Uncertainty in emission center.
                - 'fwhm_emission': Emission line full width at half maximum (FWHM).
                - 'fwhm_emission_error': Uncertainty in emission FWHM.
                - 'ew_emission': Emission line equivalent width.
                - 'ew_emission_error': Uncertainty in emission equivalent width.
                - 'rv_emission' (float): Radial velocity (RV) of the emission line in km/s.
                - 'rv_emission_error' (float): Error on the estimated RV of the emission line.
                - 'absorption_center': Absorption line center wavelength.
                - 'absorption_center_error': Uncertainty in absorption center.
                - 'fwhm_absorption': Absorption line FWHM.
                - 'fwhm_absorption_error': Uncertainty in absorption FWHM.
                - 'ew_absorption': Absorption line equivalent width.
                - 'ew_absorption_error': Uncertainty in absorption equivalent width.
                - 'rv_absorption' (float): Radial velocity (RV) of the absorption line in km/s.
                - 'rv_absorption_error' (float): Error on the estimated RV of the absorption line.

        Notes:
            The function models the spectrum as:
            continuum + emission Voigt profile + absorption Voigt profile (negative amplitude).
            Initial guesses are estimated from the input flux.
            Monte Carlo sampling perturbs the data by residual noise to estimate equivalent width uncertainties.
        """
        mask = (x >= center_line - fit_width/2) & (x <= center_line + fit_width/2)
        x_fit = x[mask]
        y_fit = y[mask]

        guesses = self._guess_initials(x_fit, y_fit)

        const_mod = ConstantModel(prefix='const_')
        em_mod = VoigtModel(prefix='em_')
        abs_mod = VoigtModel(prefix='abs_')
        model = const_mod + em_mod + abs_mod
        params = model.make_params()

        # Set continuum initial value
        params['const_c'].set(value=guesses['continuum'])
        # Emission component parameters
        params['em_amplitude'].set(value=guesses['em_amp'], min=0)
        params['em_center'].set(value=guesses['em_center'], min=center_line - fit_width/2, max=center_line + fit_width/2)
        params['em_sigma'].set(value=fit_width / 15, min=1e-4)
        params['em_gamma'].set(value=fit_width / 20, min=1e-4)
        # Absorption component parameters (amplitude negative)
        params['abs_amplitude'].set(value=-guesses['abs_amp'], max=0)
        params['abs_center'].set(value=guesses['abs_center'], min=center_line - fit_width/2, max=center_line + fit_width/2)
        params['abs_sigma'].set(value=fit_width / 15, min=1e-4)
        params['abs_gamma'].set(value=fit_width / 20, min=1e-4)

        # Perform the fit
        result = self._fit_model(x_fit, y_fit, model, params)
        comps = result.eval_components(x=x_fit)
        continuum = result.params['const_c'].value

        # Emission line parameters & errors
        em_center = result.params['em_center'].value
        em_sigma = result.params['em_sigma'].value
        em_gamma = result.params['em_gamma'].value
        em_center_err = result.params['em_center'].stderr or 0.0
        em_sigma_err = result.params['em_sigma'].stderr or 0.0
        em_gamma_err = result.params['em_gamma'].stderr or 0.0
        fwhm_em = self._voigt_fwhm(em_sigma, em_gamma)
        dfwhm_dsigma_em = (2.355**2 * em_sigma) / np.sqrt(0.2166 * (2*em_gamma)**2 + (2.355 * em_sigma)**2)
        dfwhm_dgamma_em = (0.5346 * 2 + (0.2166 * 4 * em_gamma) / (2 * np.sqrt(0.2166 * (2 * em_gamma)**2 + (2.355 * em_sigma)**2)))
        fwhm_em_err = np.sqrt((dfwhm_dsigma_em * em_sigma_err)**2 + (dfwhm_dgamma_em * em_gamma_err)**2)

        # Absorption line parameters & errors
        abs_center = result.params['abs_center'].value
        abs_sigma = result.params['abs_sigma'].value
        abs_gamma = result.params['abs_gamma'].value
        abs_center_err = result.params['abs_center'].stderr or 0.0
        abs_sigma_err = result.params['abs_sigma'].stderr or 0.0
        abs_gamma_err = result.params['abs_gamma'].stderr or 0.0
        fwhm_abs = self._voigt_fwhm(abs_sigma, abs_gamma)
        dfwhm_dsigma_abs = (2.355**2 * abs_sigma) / np.sqrt(0.2166 * (2*abs_gamma)**2 + (2.355 * abs_sigma)**2)
        dfwhm_dgamma_abs = (0.5346 * 2 + (0.2166 * 4 * abs_gamma) / (2 * np.sqrt(0.2166 * (2 * abs_gamma)**2 + (2.355 * abs_sigma)**2)))
        fwhm_abs_err = np.sqrt((dfwhm_dsigma_abs * abs_sigma_err)**2 + (dfwhm_dgamma_abs * abs_gamma_err)**2)

        model_y = result.best_fit
        model_y_err = np.std(y_fit - model_y)

        # Equivalent widths and uncertainties via Monte Carlo
        ew_em = self._compute_ew(x_fit, comps['em_'] + continuum, continuum)
        ew_em_err = self._monte_carlo_ew_error(x_fit, y_fit, model, params, 'em_', n_mc=n_mc, n_workers=n_workers)

        ew_abs = self._compute_ew(x_fit, comps['abs_'] + continuum, continuum)
        ew_abs_err = self._monte_carlo_ew_error(x_fit, y_fit, model, params, 'abs_', n_mc=n_mc, n_workers=n_workers)

        # Radial velocities
        rv_em, rv_em_err = self.compute_radial_velocity(em_center, em_center_err, rest_wavelength=center_line)
        rv_abs, rv_abs_err = self.compute_radial_velocity(abs_center, abs_center_err, rest_wavelength=center_line)

        return {
            'result': result,
            'x_fit': x_fit,
            'y_fit': y_fit,
            'model_y': model_y,
            'model_y_err': model_y_err,
            'emission_center': em_center,
            'emission_center_error': em_center_err,
            'fwhm_emission': fwhm_em,
            'fwhm_emission_error': fwhm_em_err,
            'ew_emission': ew_em,
            'ew_emission_error': ew_em_err,
            'rv_emission': rv_em,
            'rv_emission_error': rv_em_err,
            'absorption_center': abs_center,
            'absorption_center_error': abs_center_err,
            'fwhm_absorption': fwhm_abs,
            'fwhm_absorption_error': fwhm_abs_err,
            'ew_absorption': ew_abs,
            'ew_absorption_error': ew_abs_err,
            'rv_absorption': rv_abs,
            'rv_absorption_error': rv_abs_err
        }
