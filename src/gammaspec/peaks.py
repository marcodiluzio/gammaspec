"""Script to perform peak elaboration on gamma spectra

description

it contains functions:
_peak_integral, _get_fit, _manage_fit

it contains classes:
-

This module can be imported into another script with:
"from gammaspec import peaks"
giving access to all its methods and classes

author: Marco Di Luzio
email: m.diluzio@inrim.it
"""

import os
import pickle
import datetime
import numpy as np
import pandas as pd
#import asyncio
import warnings
from scipy.optimize import curve_fit, OptimizeWarning
from scipy.stats import norm, t, skewnorm, cauchy, lognorm, gamma, chi2, gennorm
from scipy.integrate import quad

warnings.simplefilter("error", OptimizeWarning)
FWHMSIG = 2.355

class SpectrumElaboration:
    def __init__(self, spectrum, regions=None, bkg=None):
        self.spc = spectrum

        #elaboration things
        if regions is not None:
            self.regions = regions
        else:
            self.regions = []
        self.bkg = bkg
        if self.bkg is not None and self.bkg.channels != self.spc.channels:
            raise ValueError('spectrum and background need to have the same number of channels')

    def _preserve(self):
        preserve_name = os.path.splitext(self.spc.o_filename)[0]
        elaborated = f'elaborated_{datetime.date.today().strftime("%Y-%m-%d")}'
        with open(f'{preserve_name}_{elaborated}.pel', 'wb') as pfile:
            pickle.dump(self, pfile)

    def _add_region(self):
        pass

    def _adjust_bkg(self):
        if self.bkg is None:
            return np.zeros(self.spc.channels)

        return np.array(self.bkg.counts/self.bkg.live_time*self.spc.live_time, dtype=int)

    def _elab_region(self):
        #init
        centroid_lst = []
        energy_lst = []
        area_lst = []
        area_unc_lst = []
        fwhm_lst = []
        fwhm_unc_lst = []
        activity_lst = []
        activity_unc_lst = []
        isoptope_ref = []
        infos = []
        for reg in self.regions:
            BKG_counts = self._adjust_bkg()
            X, Y, _ = reg.spectrum_slice(self.spc.counts, BKG_counts)

            try:
                Yvar = self.spc.var[X] + BKG_counts[X]
            except (TypeError, IndexError):
                Yvar = None

            popt, pcov = _manage_fit(X, Y, _, reg.func, Y2=Yvar)

            perr = np.sqrt(np.diag(pcov))
            spcICR, uspcICR = self.spc.input_count_rate()

            if popt[0] is not None:

                #prepare for output
                centroid_lst.append(popt[1])
                energy_lst.append(reg.energy)
                area_lst.append(popt[0])
                area_unc_lst.append(perr[0])
                fwhm_lst.append(popt[2] * FWHMSIG)
                fwhm_unc_lst.append(perr[2] * FWHMSIG)
                activity_lst.append(reg.activity)
                activity_unc_lst.append(reg.activity_unc)
                isoptope_ref.append(reg.ilabel)
                infos.append((popt, pcov, reg.func, self.spc.live_time, self.spc.live_time_unc, self.spc.real_time, self.spc.real_time_unc, spcICR, uspcICR))

        return centroid_lst, energy_lst, area_lst, area_unc_lst, fwhm_lst, fwhm_unc_lst, activity_lst, activity_unc_lst, isoptope_ref, infos


class CalibrationsProcedure:
    def __init__(self, spectrumelaboration_list):
        self.sp_list = [item for item in spectrumelaboration_list if isinstance(item, SpectrumElaboration)]
        self.centroid, self.energy, self.area, self.area_unc, self.fwhm, self.fwhm_unc, self.activity, self.activity_unc, self.isotope_ref, self.infos = self.elaborate()
        self.pileup_corrected = False

        self.E_calib = np.array([0, 1, 0])
        self.E_calib_info = {'message':'default calibration'}

        self.FWHM_calib = np.array([0, 1, 0])
        self.FWHM_calib_info = {'message':'default calibration'}

        self.EFF_calib = np.array([0, 1, 0])
        self.EFF_calib_cov = np.identity(len(self.EFF_calib))
        self.EFF_calib_info = {'message':'default calibration', 'fit':_poly_eff_fit, 'polyfit_n':4}

    def elaborate(self):
        centroid_lst, energy_lst, area_lst, area_unc_lst, fwhm_lst, fwhm_unc_lst, activity_lst, activity_unc_lst, isotope_ref, infos = [], [], [], [], [], [], [], [], [], []
        for item in self.sp_list:
            centroid_plt, energy_plt, area_plt, area_unc_plt, fwhm_plt, fwhm_unc_plt, activity_plt, activity_unc_plt, isotope_ref_plt, infos_plt = item._elab_region()

            centroid_lst += centroid_plt
            energy_lst += energy_plt
            area_lst += area_plt
            area_unc_lst += area_unc_plt
            fwhm_lst += fwhm_plt
            fwhm_unc_lst += fwhm_unc_plt
            activity_lst += activity_plt
            activity_unc_lst += activity_unc_plt
            isotope_ref += isotope_ref_plt
            infos += infos_plt

        centroid_lst = np.array(centroid_lst, dtype=float)
        energy_lst = np.array(energy_lst, dtype=float)
        area_lst = np.array(area_lst, dtype=float)
        area_unc_lst = np.array(area_unc_lst, dtype=float)
        fwhm_lst = np.array(fwhm_lst, dtype=float)
        fwhm_unc_lst = np.array(fwhm_unc_lst, dtype=float)
        activity_lst = np.array(activity_lst, dtype=int)
        activity_unc_lst = np.array(activity_unc_lst, dtype=int)
        isotope_ref = np.array(isotope_ref, dtype=object)
        infos = np.array(infos, dtype=object)

        ord_index = np.argsort(centroid_lst)

        return centroid_lst[ord_index], energy_lst[ord_index], area_lst[ord_index], area_unc_lst[ord_index], fwhm_lst[ord_index], fwhm_unc_lst[ord_index], activity_lst[ord_index], activity_unc_lst[ord_index], isotope_ref[ord_index], infos[ord_index]

    def _pileup_correction(self, two_tau=0.0, two_tau_unc=0.0):
        pup_corr_lst, upup_corr_lst = [], []
        for item in self.infos:
            spcICR, uspcICR = item[-2], item[-1]
            pup_corr, upup_corr = evaluate_pileup_correction(spcICR, uspcICR, two_tau, two_tau_unc)
            pup_corr_lst.append(pup_corr)
            upup_corr_lst.append(upup_corr)

        pup_corr_lst = np.array(pup_corr_lst, dtype=float)
        upup_corr_lst = np.array(upup_corr_lst, dtype=float)

        if not self.pileup_corrected:
            urunc = self.area_unc / self.area
            self.area = self.area * pup_corr_lst
            self.area_unc = self.area * np.sqrt(np.power(urunc, 2) * np.power(upup_corr_lst/pup_corr_lst, 2))
            self.pileup_corrected = True

        return pup_corr_lst, upup_corr_lst

    def _report(self):
        pass

    def _preserve(self, preserve_name='Calib_0'):
        with open(f'{preserve_name}.pcb', 'wb') as pfile:
            pickle.dump(self, pfile)

    def eval_energy_calibration(self):
        if len(self.centroid) < 4:
            self.E_calib_info = {**self.E_calib_info, **{'message':'calibration error - not enough data points to perform calibration'}}
            return

        W = self.centroid[:,np.newaxis]**[_ for _ in range(3)]
        self.E_calib = np.linalg.inv(W.T@W)@(W.T@self.energy)

        WL = self.centroid[:,np.newaxis]**[_ for _ in range(2)]
        parameters_linear = np.linalg.inv(WL.T@WL)@(WL.T@self.energy)

        print(parameters_linear)

        diff0 = (self.E_calib[0] - parameters_linear[0]) / parameters_linear[0]
        diff1 = (self.E_calib[1] - parameters_linear[1]) / parameters_linear[1]
        print(diff0, diff1)

        self.E_calib_info = {**self.E_calib_info, **{'message':'E calibration succesful'}}

    def eval_fwhm_calibration(self):
        if len(self.centroid) < 4:
            self.FWHM_calib_info = {**self.FWHM_calib_info, **{'message':'calibration error - not enough data points to perform calibration'}}
            return None

        #in channels convert to energy
        Y, uY = self.fwhm*self.E_calib[1], self.fwhm_unc*self.E_calib[1]
        Y = np.power(Y, 2)

        W = self.energy[:,np.newaxis]**[_ for _ in range(3)]

        self.FWHM_calib = np.linalg.inv(W.T@W)@(W.T@Y)
        self.FWHM_calib_info = {**self.FWHM_calib_info, **{'message':'E calibration succesful'}}

    def eval_efficiency_calibration(self):

        def dataframe_helper(df, isotope_label, energy_label):
            _filter_isotope = df['isotope'] == isotope_label
            _filter_energy = df['energy / keV'] == energy_label

            combined_filter = _filter_isotope & _filter_energy
            sub_df = df[combined_filter]

            return (sub_df.loc[sub_df.index[0], 'lambda / s'], sub_df.loc[sub_df.index[0], 'u_lambda / s'], sub_df.loc[sub_df.index[0], 'yield / 1'], sub_df.loc[sub_df.index[0], 'u_yield / 1'])

        def uncertainty_evaluation_expg(activity, activity_unc, times, literature_data):

            def _formula(pms):
                return (pms[0] * pms[1] * pms[2] * (1 - np.exp(-pms[3] * pms[4]))) / (pms[3] * pms[4])

            unc = []

            for A, UA, TM, LDT in zip(activity, activity_unc, times, literature_data):
                params = np.array([A, TM[0], LDT[2], LDT[0], TM[2]], dtype=float)
                unc_p = np.array([UA, TM[1], LDT[3], LDT[1], TM[3]], dtype=float)
                pcov = np.diag(np.power(unc_p,2))
                sc = []

                for idx, local_unc in enumerate(unc_p):
                    mask = np.zeros(len(unc_p))
                    mask[idx] = 1
                    p_plus = params + unc_p * mask
                    p_minus = params - unc_p * mask
                    sc.append((_formula(p_plus) - _formula(p_minus)) / (2*local_unc + 1E-24))

                sc = np.array(sc, dtype=float)
                unc.append(np.sqrt((sc@pcov)@sc))

            return np.array(unc, dtype=float)

        if len(self.centroid) < 6:
            return None

        iso_db = load_isotope_database(short=False)
        literature_data = np.array([dataframe_helper(iso_db, isotope_label, energy_label) for isotope_label, energy_label in zip(self.isotope_ref, self.energy)], dtype=float)

        times = np.array([(info[3], info[4], info[5], info[6]) for info in self.infos], dtype=float)/1E12

        expected_gammas = (self.activity * times.T[0] * literature_data.T[2] * (1-np.exp(-literature_data.T[0] * times.T[2]))) / (literature_data.T[0] * times.T[2])
        expected_gammas_unc = uncertainty_evaluation_expg(self.activity, self.activity_unc, times, literature_data)

        #efficiency
        eps = self.area / expected_gammas
        eps_unc = eps * np.sqrt(np.power(self.area_unc/self.area,2) + np.power(expected_gammas_unc/expected_gammas,2))

        for EE, EPS, UEPS in zip(self.energy, eps, eps_unc):
            print(EE, EPS, np.log(EPS), UEPS/EPS)

        Y = np.log(eps)
        uY = eps_unc / eps

        #Y = eps
        #uY = eps_unc

        print(Y)
        print(uY)

        #V = np.identity(len(eps))
        V = np.zeros((len(eps), len(eps)))
        for idx0, isot0 in enumerate(self.isotope_ref):
            for idx1, isot1 in enumerate(self.isotope_ref):
                if isot0 == isot1:
                    V[idx0, idx1] = 1 * uY[idx0] * uY[idx1]

        #EM = self.energy / 1000
        EM = np.log(self.energy)

        #W = EM[:,np.newaxis]**[_ for _ in range(4)]
        W = EM[:,np.newaxis]**[0, 1]#_ for _ in range(4)]
        self.EFF_calib = np.linalg.inv(W.T@V@W)@(W.T@V@Y)
        #parameters = np.linalg.inv(W.T@W)@(W.T@Y)#uncertainty here

        #residuals
        residuals = Y - self.EFF_calib@W.T

        #covariance matrix
        n, k = Y.shape[0], W.shape[1]
        self.EFF_calib_cov = np.true_divide(1, n-k) * (residuals.T@V@residuals) * np.linalg.inv(W.T@V@W)
        self.EFF_calib_info = {**self.EFF_calib_info, **{'message':'EFF calibration succesful'}}

    def calibration(self, two_tau=0.0, two_tau_unc=0.0):

        if not self.pileup_corrected and two_tau != 0:
            self._pileup_correction(two_tau=two_tau, two_tau_unc=two_tau_unc)

        self.eval_energy_calibration()
        self.eval_fwhm_calibration()
        self.eval_efficiency_calibration()


class Region:
    """Class to manage elaboration of region of interest in spectrum

    Attributes
    ----------
    self.low : int
        left hand side limit of ROI to consider
    self.right : int
        right hand side limit of ROI to consider
    self.centroid : int
        centroid of the peak

    Methods
    -------
    _check_input()
        assure correct input is provided to the class
    spectrum_slice()
        provide a slice for the array of the spectrum counts
    """
    def __init__(self, low_limit, high_limit, centroid=None):
        """Region class constructor

        Parameters
        ----------
        low_limit : int
            left hand side limit of ROI to consider
        high_limit : int
            right hand side limit of ROI to consider
        centroid : int
            centroid of the peak (default None)

        Return
        ------
        None
        """
        self.low, self.right, self.centroid = self._check_input(low_limit, high_limit, centroid)

    def _check_input(self, low_limit, high_limit, centroid):
        """Assure correct input is provided to the class

        Parameters
        ----------
        low_limit : int
            left hand side limit of ROI to consider
        high_limit : int
            right hand side limit of ROI to consider
        centroid : int
            centroid of the peak

        Return
        ------
        low_limit : int
            left hand side limit of ROI to consider
        high_limit : int
            right hand side limit of ROI to consider
        centroid : int
            centroid of the peak
        """
        if low_limit > high_limit - 10:
            raise ValueError('Region should be at least 10 channel long')
        if centroid is None:
            centroid = low_limit + (high_limit - low_limit)/2
        if centroid >= high_limit or centroid <= low_limit:
            raise ValueError('Centroid outside region limits')
        return int(low_limit), int(high_limit), int(centroid)

    def spectrum_slice(self, counts, bkg_counts=None):
        """Provide a slice for the array of the spectrum counts with consistent additional information

        Parameters
        ----------
        counts : np.array
            array of counts of the whole spectrum
        bkg_counts : np.array (or None)
            array of background counts to subtract to the spectrum [if None it defaults to an array of 0] (default None)

        Return
        ------
        X : np.array
            array of indices [number of channels] corresponding to the region of the spectrum to evaluate
        Y : np.array
            array of number of counts corresponding to X indices
        centroid : int
            centroid of the peak
        """
        X = np.arange(self.low, self.right, dtype=int)

        if bkg_counts is None:
            bkg_counts = np.zeros(len(counts))

        try:
            Y = counts[X] - bkg_counts[X]
            np.maximum(Y, 0, out=Y)
            Y = Y + 0.01
        except IndexError:
            return None, None, None
        return X, Y, self.centroid


class RegionElab(Region):
    """Class to manage elaboration of region of interest in spectrum (inherits from Region)

    Attributes
    ----------
    self.energy : np.array
        array of float of the energy uncertainty for the corresponding data
    self.activity : np.array
        array of int of the activity for the corresponding data
    self.activity_unc : np.array
        array of int of the activity uncertainty for the corresponding data
    self.ilabel : np.array
        array of str label of the corresponding isotope
    self._funcs : str
        albel referring to the fitting curve to use
    self.func : fitting_function
        callable function, takes X and fitting parameters and return estimate of Y
    self.funcBKG : fitting_function
        callable function, takes X and fitting parameters and return estimate of the background within Y

    Methods
    -------
    __repr__()
        overwrite the default __repr__ function to show information about the region
    """
    def __init__(self, low_limit, high_limit, centroid, energy, activity, activity_unc, ilabel, lfuncs):
        """RegionElab class constructor

        Parameters
        ----------
        low_limit : int
            left hand side limit of ROI to consider
        high_limit : int
            right hand side limit of ROI to consider
        centroid : int
            centroid of the peak
        energy : float
            energy of the emission in keV
        activity : int
            activity of the corresponding nuclide at the time of the acquisition in Bq
        activity_unc : int
            uncertianty of activity of the corresponding nuclide at the time of the acquisition in Bq
        ilabel : str
            label to identify the corresponding isotope
        lfuncs : str
            albel to identify the fitting funtion to use
            valid options are:
            'normal_step'       : gaussian function with stepped background
            'normal_slope'      : gaussian function with sloped background
            'normal_stepcurve'  : gaussian function with stepped background + background curve
            'normal_slopecurve' : gaussian function with sloped background + background curve

        Return
        ------
        None
        """
        super().__init__(low_limit, high_limit, centroid)
        self.energy = energy
        self.activity = activity
        self.activity_unc = activity_unc
        self.ilabel = ilabel
        self._funcs = lfuncs
        self.func, self.funcBKG = _get_fit(lfuncs)

    def __repr__(self):
        return f'({self.low} - {self.right}) -> {self.ilabel} @ {self.energy}'


def _poly_eff_fit():
    pass

def _cauchy_fit(x, A, mu, sigma):
    return A * (cauchy.pdf(x, loc=mu, scale=sigma))
    
def _normal_fit(x, A, mu, sigma):
    return A * (norm.pdf(x, loc=mu, scale=sigma))
    
def _student_fit(x, A, mu, sigma, df):
    return A * (t.pdf(x, df, loc=mu, scale=sigma))
    
def _leftskew(x, A, mu, sigma, LStep, LScale):
    return _normal_fit(x, A, mu, sigma) * LStep * norm.cdf((mu - x)/LScale, scale=sigma)
    
def _rightskew(x, A, mu, sigma, RStep, RScale):
    return _normal_fit(x, A, mu, sigma) * RStep * norm.cdf((x - mu)/RScale, scale=sigma)

def _CONST_BKG(BKG):
    return BKG
    
def _SLOPE(x, mu, slop):
    return 1 + slop * (x-mu)
    
def _CURVE(x, mu, a, b):
    return a * (x-mu) + b * np.power(x-mu, 2)

def _STEP(x, A, mu, sigma, step):
    return A * step * norm.cdf(mu - x, scale=sigma)

def _fit_normal_step(x, A, mu, sigma, step, BKG):
    #return A * (norm.pdf(x, loc=mu, scale=sigma)) + A * step * norm.cdf(mu - x, scale=sigma) + BKG
    return A * (norm.pdf(x, loc=mu, scale=sigma) + step * norm.cdf(mu - x, scale=sigma)) + BKG

def _fit_normal_stepBKG(x, A, mu, sigma, step, BKG):
    return A * step * norm.cdf(mu - x, scale=sigma) + BKG

def _fit_normal_slope(x, A, mu, sigma, slop, BKG):
    return A * (norm.pdf(x, loc=mu, scale=sigma)) + (1 + slop * (x-mu)) * BKG

def _fit_normal_slopeBKG(x, A, mu, sigma, slop, BKG):
    return (1 + slop * (x-mu)) * BKG

def _fit_bothskews_step(x, A, mu, sigma, LStep, LScale, RStep, RScale, step, BKG):
    return A * (norm.pdf(x, loc=mu, scale=sigma) + norm.pdf(x, loc=mu, scale=sigma) * LStep * norm.cdf((mu - x)/LScale, scale=sigma) + norm.pdf(x, loc=mu, scale=sigma) * RStep * norm.cdf((x - mu)/RScale, scale=sigma) + step * norm.cdf(mu - x, scale=sigma)) + BKG

def _fit_bothskews_step_stepBKG(x, A, mu, sigma, LStep, LScale, RStep, RScale, step, BKG):
    return A * step * norm.cdf(mu - x, scale=sigma) + BKG

def _fit_normal_stepcurve(x, A, mu, sigma, step, a, b, BKG):
    return A * ((norm.pdf(x, loc=mu, scale=sigma)) + step * norm.cdf(mu - x, scale=sigma)) + BKG + a * (x-mu) + b * np.power(x-mu, 2)

def _fit_normal_stepcurveBKG(x, A, mu, sigma, step, a, b, BKG):
    return A * step * norm.cdf(mu - x, scale=sigma) + BKG + a * (x-mu) + b * np.power(x-mu, 2)

def _fit_normal_slopecurve(x, A, mu, sigma, slop, a, b, BKG):
    return A * (norm.pdf(x, loc=mu, scale=sigma)) + (1 + slop * (x-mu)) * BKG + a * (x-mu) + b * np.power(x-mu, 2)

def _fit_normal_slopecurveBKG(x, A, mu, sigma, slop, a, b, BKG):
    return (1 + slop * (x-mu)) * BKG + a * (x-mu) + b * np.power(x-mu, 2)

def _fit_gennorm(x, A, mu, sigma, mbeta, beta, step, BKG):
    return A * (norm.pdf(x, loc=mu, scale=sigma) + mbeta * gennorm.pdf(x, beta, loc=mu, scale=1/np.sqrt(2)) + step * norm.cdf(mu - x, scale=sigma)) + BKG

def _fit_gennormBKG(x, A, mu, sigma, mbeta, beta, step, BKG):
    return A * step * norm.cdf(mu - x, scale=sigma) + BKG

def _peak_integral(spectrum, cond_ROI, cond_BKG_CH):
    """Peak area evaluation with the classic gross integral method

    Parameters
    ----------
    spectrum : spectra.GammaSpectrum
        spectrum object
    cond_ROI : tuple
        2-item tuple indicating the limits of the ROI to calculate the integral
    cond_BKG_CH : tuple
        2-item tuple indicating the the number of channels (internal to the ROI) to calculate the integral on the background

    Return
    ------
    peak_integral : float
        net peak area
    peak_uncertainty : float
        uncertainty of net peak area
    """
    bkg_left = spectrum.defined_spectrum_integral(start=cond_ROI[0], end=cond_ROI[0]+cond_BKG_CH[0]) / cond_BKG_CH[0]
    bkg_right = spectrum.defined_spectrum_integral(start=cond_ROI[1]-cond_BKG_CH[1], end=cond_ROI[1]) / cond_BKG_CH[1]
    N = cond_ROI[1]-cond_BKG_CH[1] - (cond_ROI[0]+cond_BKG_CH[0])
    g_peak = spectrum.defined_spectrum_integral(start=cond_ROI[0]+cond_BKG_CH[0], end=cond_ROI[1]-cond_BKG_CH[1])
    c_peak = g_peak - np.average([bkg_left, bkg_right]) * N
    ulin = np.sqrt(g_peak + np.average([bkg_left, bkg_right]) * N)
    return c_peak, ulin/c_peak*100

def _get_fit(fit_string):
    """Get fit functions

    Parameters
    ----------
    fit_string : str
        string referring to a defined fit

    Return
    ------
    peak_fitting_function : fitting_function
        callable function, takes X and fitting parameters and return estimate of Y
    background_fitting_function : fitting_function
        callable function, takes X and fitting parameters and return estimate of the background within Y
    """
    choice = {'normal_step' : (_fit_normal_step, _fit_normal_stepBKG),
              'normal_slope' : (_fit_normal_slope, _fit_normal_slopeBKG),
              'normal_stepcurve' : (_fit_normal_stepcurve, _fit_normal_stepcurveBKG),
              'normal_slopecurve' : (_fit_normal_slopecurve, _fit_normal_slopecurveBKG),}
    return choice.get(fit_string, (_fit_normal_step, _fit_normal_stepBKG))

def _fit_unc(_fit, X, popt, pcov):
    """Calculate uncertainty of the Y estimate resulting from the fit using linearly approximated sensitivity coefficients

    Parameters
    ----------
    _fit : fitting_function
        callable function, takes X and fitting parameters and return estimate of Y
    X : np.array
        array of indices [number of channels] corresponding to the region of the spectrum to evaluate
    popt : np.array
        array of the optimized parameters for the fit
    pcov : np.array
        2d-array corresponding to the covariance matrix

    Return
    ------
    unc_fit : np.array
        array of uncertainty for each x value in X
    """
    perr = np.sqrt(np.diag(pcov))

    unc_fit = []

    for x in X:
        sc = []#sensitivity coefficients
        for idx, local_unc in enumerate(perr):
            mask = np.zeros(len(perr))
            mask[idx] = 1
            p_plus = popt + perr * mask
            p_minus = popt - perr * mask
            sc.append((_fit(x, *p_plus) - _fit(x, *p_minus)) / (2*local_unc + 1E-24))
        sc = np.array(sc, dtype=float)
        unc_fit.append(np.sqrt((sc@pcov)@sc))

    return np.array(unc_fit, dtype=float)

def _manage_fit(X, Y, centroid, s_fit, Y2=None):
    """Perform non-linear peak fitting and manage input parameters

    Parameters
    ----------
    X : np.array
        array of indices [number of channels] corresponding to the region of the spectrum to evaluate
    Y : np.array
        array of number of counts corresponding to X indices
    centroid : int
        centroid where the peak is suppposed to lie
    s_fit : fitting_function
        callable function, takes X and fitting parameters and return estimate of Y
    Y2 : np.array (or None)
        variance of Y, if None fallback to np.sqrt(Y) (default None)

    Return
    ------
    popt : np.array
        array of the optimized parameters for the fit
    pcov : np.array
        2d-array corresponding to the covariance matrix
    """

    try:
        yunc = np.sqrt(Y2)
    except (TypeError, ValueError):
        yunc = np.sqrt(Y)
    np.maximum(yunc, 1, out=yunc)

    if centroid >= np.max(X) or centroid <= np.min(X):
        centroid = np.min(X) + (np.max(X) - np.min(X))//2

    HALFWIDTH = (np.max(X) - np.min(X))/2

    bound_choice = {_fit_normal_step : ([100, centroid, 3, 0.01, 10], ((10, centroid-3, 1, -np.inf, 0.01), (np.inf, centroid+3, HALFWIDTH, np.inf, np.inf))),
                    _fit_normal_slope : ([100, centroid, 3, 0, 10], ((10, centroid-3, 1, -10, 0.01), (np.inf, centroid+3, HALFWIDTH, 10, np.inf))),
                    _fit_normal_stepcurve : ([100, centroid, 3, 0.01, 0, 0, 10], ((10, centroid-3, 1, -np.inf, -np.inf, -np.inf, 0.01), (np.inf, centroid+3, HALFWIDTH, np.inf, np.inf, np.inf, np.inf))),
                    _fit_normal_slopecurve : ([100, centroid, 3, 0, 0, 0, 10], ((10, centroid-3, 1, -np.inf, -100, -10, 0.01), (np.inf, centroid+3, HALFWIDTH, np.inf, 100, 10, np.inf))),
                    }

    P0, BOUNDS = bound_choice.get(s_fit, (None, None))

    if P0 is None:
        return [None, None, None, 0.0, 0.0], [1.0]

    try:
        popt, pcov, moreinfo, mesg, alert = curve_fit(s_fit, X, Y, p0=P0, sigma=yunc, bounds=BOUNDS, full_output=True, absolute_sigma=True)
    except (ValueError, RuntimeError, RuntimeWarning, ValueError, OptimizeWarning) as error:
        popt, pcov, moreinfo, mesg, alert = [None, None, None, 0.0, 0.0], [1.0], {}, '', ''
    return popt, pcov

def evaluate_pileup_correction(ICR, uICR, two_tau, utwo_tau):
    """Calculate pileup correction factor based on input count rate and 2t parameter as exp(ICR * 2t)

    Parameters
    ----------
    ICR : float
        descr
    uICR : float
        descr
    two_tau : float
        descr
    utwo_tau : float
        descr

    Return
    ------
    pileup_correction : float
        pileup_correction
    pileup_correction_unc : float
        standard uncertainty of pileup_correction_unc
    """
    #np.exp(ICR * two_tau)
    sc = np.array([(np.exp((ICR+uICR) * two_tau) - np.exp((ICR-uICR) * two_tau))/(2*uICR + 1E-24), (np.exp(ICR * (two_tau+utwo_tau)) - np.exp(ICR * (two_tau-utwo_tau)))/(2*utwo_tau + 1E-24)])
    unc = np.array([uICR, utwo_tau])

    return np.exp(ICR * two_tau), np.sqrt(np.sum(np.power(sc*unc, 2)))

def load_isotope_database(filename='isotope_database.xlsx', short=False):
    """Recall database with information about relevant isotopes for calibration and other calculations

    Parameters
    ----------
    filename : str
        excel file containing the relevant information (default 'isotope_database.xlsx')
    short : bool
        return a condensed version of the dataframe with only isotope name and energy (default False)

    Return
    ------
    df : pd.DataFrame
        database with isotope information
    """
    df = pd.read_excel(filename, sheet_name=0, header=0, names=('isotope', 'energy / keV', 'lambda / s', 'u_lambda / s', 'reference lambda', 'yield / 1', 'u_yield / 1', 'reference yield'))

    if short:
        return df[['isotope', 'energy / keV']]

    return df

def _update_activities(sdb):
    #check halflives
    _filnan = sdb['l / s-1'].isna()
    if np.sum(_filnan) > 0:
        idb = load_isotope_database()

        lbds, ulbds = [], []

        for item in sdb['isotope']:
            _filter = idb['isotope'] == item
            try:
                lbd = idb[_filter].loc[idb[_filter].index[0], 'lambda / s']
                ulbd = idb[_filter].loc[idb[_filter].index[0], 'u_lambda / s']
            except IndexError:
                lbd, ulbd = np.nan, np.nan
            lbds.append(lbd)
            ulbds.append(ulbd)
        lbds = np.array(lbds, dtype=float)
        ulbds = np.array(ulbds, dtype=float)
        sdb['l / s-1'] = lbds
        sdb['ul / s-1'] = ulbds

    #update activities
    sdb['curr date'] = datetime.datetime.today()
    td = sdb['curr date'] - sdb['ref date']
    td = td.dt.total_seconds()
    sdb['curr A / Bq'] = sdb['A / Bq'] * np.exp(-sdb['l / s-1'] * td)

    cs1 = np.exp(-sdb['l / s-1'] * td)
    cs2 = -td * np.exp(-sdb['l / s-1'] * td)
    cs3 = -sdb['l / s-1'] * np.exp(-sdb['l / s-1'] * td)

    udt = sdb['A / Bq'] * 0 + 60

    pw1 = np.power(cs1 * sdb['uA / Bq'], 2)
    pw2 = np.power(cs2 * sdb['ul / s-1'], 2)
    pw3 = np.power(cs3 * udt, 2)
    sdb['curr uA / Bq'] = np.sqrt(pw1 + pw2 + pw3)

    return sdb

def load_source_database(filename='Activity_table.pac'):
    with open(filename, 'rb') as idb:
        sdb = pickle.load(idb)

    sdb = _update_activities(sdb)

    return sdb
    
if __name__ == '__main__':
    pass
