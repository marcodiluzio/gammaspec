"""Script to perform peak elaboration on gamma spectra

description

it contains functions:
-

it contains classes:
PeakFit, Region, Elaboration

This module can be imported into another script with:
"from gammaspec import peaks"
giving access to all its methods and classes

author: Marco Di Luzio
email: m.diluzio@inrim.it
"""


import numpy as np
import matplotlib.pyplot as plt
import asyncio
import warnings
from scipy.optimize import curve_fit, OptimizeWarning
from scipy.stats import norm, t, skewnorm, cauchy, lognorm, gamma
from scipy.integrate import quad
import spectra as gspec
import automated_identification_photopeaks as aip

warnings.simplefilter("error", OptimizeWarning)
FWHMSIG = 1.665

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


class PeakFit:
    """Class for base peak fit
    
    Attributes
    ----------
    self.name : str
        name identifying the peak fitting
    self.P0 : list
        descr
    self.BOUNDS : list
        descr
    self._PEAK : list
        descr
    self._BACKGROUND : int
        descr
    self.coeff : np.array()
        descr
    self.var_matrix : np.array()
        descr
    
    Methods
    -------
    __repr__()
        descr
    _fit()
        descr
    calculate_area()
        descr
    evaluate()
        evaluate the fit
    coro_evaluate()
        coroutine wrapper for the evaluate function to improve performance
    _draw_total()
        draw total line (peaks + background), for plotting
    _draw_partials()
        draw peak lines, for plotting
    _draw_background()
        draw background line, for plotting
    """
    def __init__(self, **kwargs):
        self.name = 'Base Peak Fit'
        self.P0 = []
        self.BOUNDS = (None,None)
        self._PEAK = [1]
        self._BACKGROUND = 1

        self.coeff = [0]
        self.var_matrix = [0]
        self.centroids = [0]
        self.u_centroids = [0]
        self.peaks = [0]
        self.ur_peaks = [0]
        self.intmsg = -1
        self.chisq = 9999
        self.settings = {'max_chisq':kwargs.get('max_chisq', 10), 'bound_peak':kwargs.get('bound_peak', 2), 'bound_fwhm':kwargs.get('bound_fwhm', 2)}
        #self._parameter_descriptor = ()#10
        
    def __repr__(self):
        return f'{self.name}'
        
    def _fit(self, x, *p):
        """placeholder"""
        return None
        
    def calculate_area_and_centroids(self, X):
        return np.array([0]), np.array([0]), np.array([0]), np.array([1])
        
    async def coro_evaluate(self, X, Y):
        self.evaluate(X, Y)
        return self
        
    def evaluate(self, X, Y):
        try:
            popt, pcov, moreinfo, mesg, alert = curve_fit(self._fit, X, Y, p0=self.P0, sigma=np.sqrt(Y), 
                            bounds=self.BOUNDS, full_output=True,
                            absolute_sigma=True)
        except (ValueError, RuntimeError, RuntimeWarning, ValueError, OptimizeWarning) as error:
            popt, pcov, moreinfo, mesg, alert = [0.0], [0.0], {}, '', -1
        if alert in (1, 2, 3, 4):
            self.coeff = popt
            self.var_matrix = pcov
            self.centroids, self.u_centroids, self.peaks, self.ur_peaks = self.calculate_area_and_centroids(X)            
            res = Y - self._draw_total(X)
            self.chisq = np.sum(np.power(res, 2)/Y) / (len(Y) - len(self.coeff))

            #checks here or later? we'll see
            if self.chisq < self.settings['max_chisq']:
                self.intmsg = 0
            else:
                self.intmsg = 1
        else:
            self.intmsg = -1
            
    def _draw_total(self, x):
        self._PEAK = [1] * len(self._PEAK)
        self._BACKGROUND = 1
        return self._fit(x, *self.coeff)
    
    def _draw_partials(self, x):
        if len(self._PEAK) > 1:
            self._BACKGROUND = 1

            fits = []
            for _idx in range(len(self._PEAK)):
                self._PEAK = [0] * len(self._PEAK)
                self._PEAK[_idx] = 1
                fits.append(self._fit(x, *self.coeff))
            return fits
        return []

    def _draw_background(self, x):
        self._PEAK = [0] * len(self._PEAK)
        self._BACKGROUND = 1
        return self._fit(x, *self.coeff)
        
    def _calculate_peak_area(self, X):
        self._PEAK = [0] * len(self._PEAK)
        self._BACKGROUND = 0
        peakareas = []
        uncertainties = []
        
        for idx, npeak in enumerate(self.centroids):
            self._PEAK[idx] = 1
            area_calculation_results = quad(self._fit, np.min(X), np.max(X), args=tuple(self.coeff))
            peakareas.append(area_calculation_results[0])
            cs = []
            copycoeff = np.copy(self.coeff)
            for nn, value in enumerate(np.sqrt(np.diag(self.var_matrix))):
                copycoeff[nn] = self.coeff[nn] + value
                Psl = quad(self._fit, np.min(X), np.max(X), args=tuple(copycoeff))
                copycoeff[nn] = self.coeff[nn] - value
                Msl = quad(self._fit, np.min(X), np.max(X), args=tuple(copycoeff))
                copycoeff[nn] = self.coeff[nn]
                cs.append((Psl[0] - Msl[0])/(2 * value + 1E-24))
            cs = np.array(cs)
            uncertainties.append(np.sqrt((cs.T@self.var_matrix)@cs))
            self._PEAK[idx] = 0
        peakareas = np.array(peakareas)
        rel_uncertainties = np.array(uncertainties) / peakareas
        return peakareas, rel_uncertainties


class StudentFit(PeakFit):
    """Subclass of PeakFit
    
    Attributes
    ----------
    
    Methods
    -------
    _fit()
        overwrites the _fit() method of PeakFit
    calculate_area_and_centroids()
        overwrites the calculate_area_and_centroids() method of PeakFit
    """
    def __init__(self, Region, cal_fwhm, **kwargs):
        super().__init__(**kwargs)
        self.name = 't-1PEAK (), BKG ()'
        
        _cd = self.settings['bound_peak']
        _fd = self.settings['bound_fwhm']

        FWHM1 = np.sqrt(Region.centroids[0] * cal_fwhm[0] + cal_fwhm[1])
        self.P0 = [100, Region.centroids[0], FWHM1/FWHMSIG, 5, 1]        
        self.BOUNDS = ((5, Region.centroids[0] - FWHM1/_cd, FWHM1/(_fd*FWHMSIG), 2, 0.01), (np.inf, Region.centroids[0] + FWHM1/_cd, FWHM1*_fd/FWHMSIG, np.inf, np.inf))
        
    def _fit(self, x, *p):
        A, mu, sigma, df, BKG = p
        return (_student_fit(x, A, mu, sigma, df)) * self._PEAK[0] + (_CONST_BKG(BKG)) * self._BACKGROUND
        
    def calculate_area_and_centroids(self, X):
        uu = np.sqrt(np.diag(self.var_matrix))
        return np.array(self.coeff[1:2], dtype=float), np.array(uu[1:2], dtype=float), np.array(self.coeff[0:1], dtype=float), np.array(uu[0:1]/self.coeff[0:1], dtype=float)


class StudentFit_Step(PeakFit):
    """Subclass of PeakFit
    
    Attributes
    ----------
    
    Methods
    -------
    _fit()
        overwrites the _fit() method of PeakFit
    calculate_area_and_centroids()
        overwrites the calculate_area_and_centroids() method of PeakFit
    """
    def __init__(self, Region, cal_fwhm, **kwargs):
        super().__init__(**kwargs)
        self.name = 't-1PEAK (), BKG (ST)'

        _cd = self.settings['bound_peak']
        _fd = self.settings['bound_fwhm']

        FWHM1 = np.sqrt(Region.centroids[0] * cal_fwhm[0] + cal_fwhm[1])
        self.P0 = [100, Region.centroids[0], FWHM1/FWHMSIG, 5, 1, 0.01]        
        self.BOUNDS = ((5, Region.centroids[0] - FWHM1/_cd, FWHM1/(_fd*FWHMSIG), 2, 0.01, 0), (np.inf, Region.centroids[0] + FWHM1/_cd, FWHM1*_fd/FWHMSIG, np.inf, np.inf, 1))

    def _fit(self, x, *p):
        A, mu, sigma, df, BKG, step = p
        return (_student_fit(x, A, mu, sigma, df)) * self._PEAK[0] + (_CONST_BKG(BKG) + _STEP(x, A, mu, sigma, step)) * self._BACKGROUND
        
    def calculate_area_and_centroids(self, X):
        uu = np.sqrt(np.diag(self.var_matrix))
        return np.array(self.coeff[1:2], dtype=float), np.array(uu[1:2], dtype=float), np.array(self.coeff[0:1], dtype=float), np.array(uu[0:1]/self.coeff[0:1], dtype=float)


class StudentFitRS_Step(PeakFit):
    """Subclass of PeakFit
    
    Attributes
    ----------
    
    Methods
    -------
    _fit()
        overwrites the _fit() method of PeakFit
    calculate_area_and_centroids()
        overwrites the calculate_area_and_centroids() method of PeakFit
    """
    def __init__(self, Region, cal_fwhm, **kwargs):
        super().__init__(*kwargs)
        self.name = 't-1PEAK (RS), BKG (ST)'

        _cd = self.settings['bound_peak']
        _fd = self.settings['bound_fwhm']

        FWHM1 = np.sqrt(Region.centroids[0] * cal_fwhm[0] + cal_fwhm[1])
        self.P0 = [100, Region.centroids[0], FWHM1/FWHMSIG, 5, 0.1, 1, 1, 0.01]        
        self.BOUNDS = ((5, Region.centroids[0] - FWHM1/_cd, FWHM1/(_fd*FWHMSIG), 2, 0, 0.5, 0.01, 0), (np.inf, Region.centroids[0] + FWHM1/_cd, FWHM1*_fd/FWHMSIG, np.inf, 2, 1000, np.inf, 1))

    def _fit(self, x, *p):
        A, mu, sigma, df, RStep, RScale, BKG, step = p
        return (_student_fit(x, A, mu, sigma, df) + _rightskew(x, A, mu, sigma, RStep, RScale)) * self._PEAK[0] + (_CONST_BKG(BKG) + _STEP(x, A, mu, sigma, step)) * self._BACKGROUND
        
    def calculate_area_and_centroids(self, X):
        uu = np.sqrt(np.diag(self.var_matrix))
        peaks, ur_peaks = self._calculate_peak_area(X)
        print('ANALYTIC', np.array([self.coeff[0]*(1 + self.coeff[4])], dtype=float), np.array([1], dtype=float))
        print('NUMERIC', peaks, ur_peaks)
        return np.array(self.coeff[1:2], dtype=float), np.array(uu[1:2], dtype=float), np.array(peaks, dtype=float), np.array(ur_peaks, dtype=float)


class StudentFitLS_Step(PeakFit):
    """Subclass of PeakFit
    
    Attributes
    ----------
    
    Methods
    -------
    _fit()
        overwrites the _fit() method of PeakFit
    calculate_area_and_centroids()
        overwrites the calculate_area_and_centroids() method of PeakFit
    """
    def __init__(self, Region, cal_fwhm, **kwargs):
        super().__init__(**kwargs)
        self.name = 't-1PEAK (LS), BKG (ST)'

        _cd = self.settings['bound_peak']
        _fd = self.settings['bound_fwhm']

        FWHM1 = np.sqrt(Region.centroids[0] * cal_fwhm[0] + cal_fwhm[1])
        self.P0 = [100, Region.centroids[0], FWHM1/FWHMSIG, 5, 0.1, 1, 1, 0.01]        
        self.BOUNDS = ((5, Region.centroids[0] - FWHM1/_cd, FWHM1/(_fd*FWHMSIG), 2, 0, 0.5, 0.01, 0), (np.inf, Region.centroids[0] + FWHM1/_cd, FWHM1*_fd/FWHMSIG, np.inf, 2, 1000, np.inf, 1))

    def _fit(self, x, *p):
        A, mu, sigma, df, LStep, LScale, BKG, step = p
        return (_student_fit(x, A, mu, sigma, df) + _leftskew(x, A, mu, sigma, LStep, LScale)) * self._PEAK[0] + (_CONST_BKG(BKG) + _STEP(x, A, mu, sigma, step)) * self._BACKGROUND
        
    def calculate_area_and_centroids(self, X):
        uu = np.sqrt(np.diag(self.var_matrix))
        peaks, ur_peaks = self._calculate_peak_area(X)
        print('ANALYTIC', np.array([self.coeff[0]*(1 + self.coeff[4])], dtype=float), np.array([1], dtype=float))
        print('NUMERIC', peaks, ur_peaks)
        return np.array(self.coeff[1:2], dtype=float), np.array(uu[1:2], dtype=float), np.array(peaks, dtype=float), np.array(ur_peaks, dtype=float)


class StudentFitBS_Step(PeakFit):
    """Subclass of PeakFit
    
    Attributes
    ----------
    
    Methods
    -------
    _fit()
        overwrites the _fit() method of PeakFit
    calculate_area_and_centroids()
        overwrites the calculate_area_and_centroids() method of PeakFit
    """
    def __init__(self, Region, cal_fwhm, **kwargs):
        super().__init__(**kwargs)
        self.name = 't-1PEAK (LS RS), BKG (ST)'

        _cd = self.settings['bound_peak']
        _fd = self.settings['bound_fwhm']

        FWHM1 = np.sqrt(Region.centroids[0] * cal_fwhm[0] + cal_fwhm[1])
        self.P0 = [100, Region.centroids[0], FWHM1/FWHMSIG, 5, 0.1, 1, 0.1, 1, 1, 0.01]        
        self.BOUNDS = ((5, Region.centroids[0] - FWHM1/_cd, FWHM1/(_fd*FWHMSIG), 2, 0, 0.5, 0, 0.5, 0.01, 0), (np.inf, Region.centroids[0] + FWHM1/_cd, FWHM1*_fd/FWHMSIG, np.inf, 2, 1000, 2, 1000, np.inf, 1))

    def _fit(self, x, *p):
        A, mu, sigma, df, LStep, LScale, RStep, RScale, BKG, step = p
        return (_student_fit(x, A, mu, sigma, df) + _leftskew(x, A, mu, sigma, LStep, LScale) + _rightskew(x, A, mu, sigma, RStep, RScale)) * self._PEAK[0] + (_CONST_BKG(BKG) + _STEP(x, A, mu, sigma, step)) * self._BACKGROUND
        
    def calculate_area_and_centroids(self, X):
        uu = np.sqrt(np.diag(self.var_matrix))
        peaks, ur_peaks = self._calculate_peak_area(X)
        return np.array(self.coeff[1:2], dtype=float), np.array(uu[1:2], dtype=float), np.array(peaks, dtype=float), np.array(ur_peaks, dtype=float)


class Region:
    """Class identifying a region of counts on the spectrum where it is suspected to be a peak
    
    Attributes
    ----------
    self.low_limit : int
        descr
    self.high_limit : int
        descr
    self.centroids : list
        descr
    self._fits : list
        descr
    #self.checks : dict
    #    descr
    #self.peak_kwargs : dict
    #    descr
    self.fit : PeakFit (or None)
        descr
    
    Methods
    -------
    _select_fit()
        descr
    peak_elaboration()
        descr
    """
    def __init__(self, low_limit, high_limit, centrs, min_netarea=100, max_allowed_unc=40, max_allowed_centroid_unc=2, max_allowed_chisq=5, left_skew=True, right_skew=True, slope=True, step=True):
        self.low_limit = low_limit
        self.high_limit = high_limit
        self.centroids = centrs
        self._fits = []
        self.checks = {'min_netarea':min_netarea, 'max_allowed_unc':max_allowed_unc, 'max_allowed_centroid_unc':max_allowed_centroid_unc, 'max_allowed_chisq':max_allowed_chisq}
        self.peak_kwargs = {'left_skew':left_skew, 'right_skew':right_skew, 'slope':slope, 'step':step}
        self.fit = self._select_fit()
        
    def _select_fit(self, idx=0):
        """Return important information about the spectrum
    
        Parameters
        ----------
        idx : int
            index within self._fits (default 0, meaning the best fit since items within self._fits are ordered from best to worst)
        
        Return
        ------
        selected_fit : PeakFit (or None)
            fit corresponding at index idx within self._fits
        """
        try:
            return self._fits[idx]
        except IndexError:
            return None
        
    async def peak_elaboration(self, counts, cal_fwhm):
        """Perform peak fitting coroutine within the region updating objects self._fits and self.fit
    
        Parameters
        ----------
        counts : np.array
            descr
        cal_fwhm : np.array
            descr
        
        Return
        ------
        None
        """
        lgt = len(self._fits)
        for _ in range(lgt):
            self._fits.pop(0)

        fit_functions_single = (StudentFit(self, cal_fwhm), StudentFit_Step(self, cal_fwhm), StudentFitRS_Step(self, cal_fwhm), StudentFitLS_Step(self, cal_fwhm), StudentFitBS_Step(self, cal_fwhm))
        X = np.arange(self.low_limit, self.high_limit+1)
        Y = counts[X]
        
        if len(self.centroids) == 1:
            fit_functions = fit_functions_single
        
        else:
            fit_functions = ()
        
        async with asyncio.TaskGroup() as taskgroup:
            collect = [taskgroup.create_task(_fit.coro_evaluate(X, Y)) for _fit in fit_functions]
        collect = [task.result() for task in collect]
        #filter
        self._fits = sorted([item for item in collect if item.intmsg == 0], key=lambda x : np.abs(x.chisq - 1))
        self.fit = self._select_fit()


class Elaboration:
    """Class for spectrum elaboration
    
    Attributes
    ----------
    self.spectrum_name : str
        descr
    self.spectrum_profile : np.array
        descr
    self.calname : str
        descr
    self.cal_energy : np.array
        descr
    self.cal_fwhm : np.array
        descr
    self.K : float
        descr
    self.regions : list
        descr
    
    Methods
    -------
    _get_spectrum_information()
        descr
    _get_calibration_information()
        descr
    """
    def __init__(self, spectrum, calibration, K=3.5):
        self.spectrum_name, self.spectrum_profile = self._get_spectrum_information(spectrum)
        self.calname, self.cal_energy, self.cal_fwhm = self._get_calibration_information(calibration)
        self.K = K
        self.regions = []

        if self.spectrum_profile is None:
            raise ValueError
            
    def _get_spectrum_information(self, spectrum):
        """Return important information about the spectrum
    
        Parameters
        ----------
        spectrum : str (path)
            name of the gamma spectrum in list mode file
        
        Return
        ------
        filename : str
            file path of the spectrum
        counts : np.array
            array of counts of the spectrum
        """
        return spectrum.filename(), spectrum.counts
        
    def _get_calibration_information(self, calibration):
        """Return important information about the calibration
    
        Parameters
        ----------
        calibration : str (path)
            name of the gamma spectrum in list mode file
        
        Return
        ------
        name : str
            file path of the calibration
        energy_parameters : np.array
            2-float array of coefficients for energy parameters
        fwhm_parameters : np.array
            2-float array of coefficients for fwhm parameters
        """
        return calibration.name, calibration.energy_params, calibration.fwhm_params
            
    #####CONTINUE
    
def mcmc_peak_search(counts, step=100, iterations=50000):
    chain = []
    
    secdif = aip.second_diff(counts, 1, 5)
    sd_secdif = aip.standard_dev_S(counts, 1, 5)
    
    filt = (secdif < 0) & (np.abs(secdif) > 2 * sd_secdif)
    current = np.arange(0,len(counts))[filt]
    print(current)
    
    counts = counts + 1
    #current = np.arange(0 + step, len(counts), step)
    #current = np.random.randint(0, len(counts), size=int(len(counts)/100))
    #initial = current
    #ame_state = 0
    plt.plot(current, np.zeros(len(current)), 'ko')
    for _ in range(iterations):
        # Propose a new sample
        #proposal = np.random.normal(current, 1).astype(int)
        proposal = np.random.uniform(low=current - 1, high=current + 2).astype(int)
        #proposal = int(np.random.uniform(low=current - 1, high=current + 2))
        proposal = np.where(proposal < 0, 0, proposal)
        proposal = np.where(proposal >= len(counts), len(counts)-1, proposal)
        #if proposal < 0:
        #    proposal = 0
        #elif proposal > len(counts):
        #    proposal = len(counts)

        # Acceptance probability
        acceptance_ratio = counts[proposal] / counts[current]

        # Accept or reject the proposal
        acceptance_value = np.random.rand(len(acceptance_ratio))#len(acceptance_ratio))
        current = np.where(acceptance_value < acceptance_ratio, proposal, current)
        #if acceptance_value < acceptance_ratio:
        #    current = proposal

        #chain.append(current)
        #plt.plot(current, [_]*len(current), 'r-')

    #print(current, initial, current - initial)
    plt.plot(current, np.ones(len(current)), 'ro')
    plt.show()
        
    return current
    
if __name__ == '__main__':
    #ccdf = gspec.ElaborationSpectrum('H:/code/gamma/elaboration/SDataR_Ho-166m_IRA_1_UG_2026-01-08_2026-02-25 12_14_27_EDT_25.asc', gspec.Calibration(), K=10)
    pass
    #peaks = mcmc_peak_search(ccdf.counts)
    #peaks = sorted(set(peaks))
    #print(len(peaks))
    #plt.plot(ccdf.counts, 'k-')
    #plt.plot(peaks, ccdf.counts[peaks], 'ro')
    #plt.show()
    
    #css = ccdf._centroid_search([184])
    #ccdf.elaboration._add_region(css)
    #ccdf.elaboration.regions[0].elaboration_fit(ccdf.counts, Calibration().fwhm_params)
    """
    reg = Region(6950, 7125, [7040])
    asyncio.run(reg.peak_elaboration(ccdf.counts, ccdf.elaboration.cal_fwhm))
    print(reg.fit)
    if reg.fit is not None:
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, height_ratios=[2,1])
        
        xplot = np.arange(reg.low_limit, reg.high_limit+1)
        ax1.bar(xplot, ccdf.counts[xplot], width=1, align='edge', alpha=0.6)
        xfit = np.linspace(np.min(xplot), np.max(xplot), 500)
        yfit = reg.fit._draw_total(xfit)
        ybkg = reg.fit._draw_background(xfit)
        ax1.plot(xfit, yfit, 'r-')
        ax1.plot(xfit, ybkg, 'k-')
        res = (ccdf.counts[xplot] - reg.fit._draw_total(xplot)) / np.sqrt(ccdf.counts[xplot])
        ax2.plot(xplot, res, 'r-o')
        
        ax1.set_xlim(np.min(xplot), np.max(xplot))
        ax1.set_ylim(1E3, None)
        ax1.set_yscale('log')
        
        fig.tight_layout()
        plt.show()"""