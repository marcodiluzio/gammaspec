"""Module to perform elaborations on gamma list mode spectra

description

it contains functions:
open_digitized_spectrumPHA, openASCfile, open_n42_file, open_events_file, _make_spectrum, _produce_decay_dataframe, safelim, retrieve_date, manage_stream_digitized_data, manage_stream_digitized_data_time, get_spectra_filenames, spectrum_multielab, multi_spectrum_elab

it contains classes:
GammaSpectrum, SpectrumChunkPHA

This module can be imported into another script with:
"from gammaspec import spectra"
giving access to all its methods and classes

author: Marco Di Luzio
email: m.diluzio@inrim.it
"""

#imports
import os
import pickle
import datetime
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import peaks as pk_elab


class GammaSpectrum:
    """Class for reconstructed gamma spectrum
    
    Attributes
    ----------
    self.real_time : int
        real time of the spectrum in ps
    self.live_time : int
        live time of the spectrum in ps
    self.real_time_unc : int
        real time uncertainty of the spectrum in ps
    self.live_time_unc : int
        live time uncertainty of the spectrum in ps
    self.channels : int
        number of channels of the spectrum (property, cannot be modified after initialized)
    self.counts : np.array
        array of int identifying counts of the spectrum for each channel
    self.extended_dead_time : str
        extended dead time in microseconds; if '-1' info not available
    self.chunk_info : list
        list with relevant information for any chunk in which the original list mode file has been divided
    self.datetime : datetime.datetime
        time of start acquisition (default None)
    self.E_cal : np.array
        array of parameters for the energy calibration in increasing power order (0,1,2) (default None)
    self.FWHM_cal : np.array
        array of parameters for the FWHM calibration in increasing power order (0,1,2) (default None)
    self.exp_params : dict
        dictionary including instrumental settings and parameters used for the acquisition of this spectrum
    self.var : np.array (or None)
        array of int identifying variance for each channel, when None the variance corresponds with np.sqrt(counts) as per Poisson statistics (default None)
    self.o_filename : str
        original filename of the list mode file from which this spectrum object is created or resulting file name if list file is elaborated with "manage_stream_digitized_data" functions
    
    Methods
    -------
    __add__()
        perform sum of spectra, overrides the defult function
    _save()
        save class information into a text file
    _saven42()
        save class information into a N42 formatted file
    _preserve()
        save class information as a pickle serialized object
    _get_chunks_info()
        return information of spectrum chunks as a pd.DataFrame
    defined_spectrum_integral()
        calculate the integral of a defined spectrum slice
    input_count_rate()
        return the value for input count rate based on total number of recorded triggers and live time
    """
    def __init__(self, real_time=0, live_time=0, counts=None, real_time_unc=0, live_time_unc=0, channels=16384, chunk_info=None, extended_dead_time='-1', var=None):
        """GammaSpectrum class constructor
        
        Parameters
        ----------
        real_time : int
            real time of the spectrum in ps (default 0)
        live_time : int
            live time of the spectrum in ps (default 0)
        counts : np.array
            array of int identifying the number of counts for any ADC channel, if None an array of zeros is created (default None)
        real_time_unc : int
            real time standard uncertainty in ps (default 0)
        live_time_unc : int
            live time standard uncertainty in ps (default 0)
        channels : int
            number of channels of spectrum (default 16384)
        chunk_info : list
            list with relevant information for any chunk in which the original list mode file has been divided
        extended_dead_time : str
            extended dead time in microseconds (default '-1')
            
        Return
        ------
        None        
        """        
        self.real_time = real_time
        self.live_time = live_time
        
        if int(np.abs(channels)) in [2**i for i in range(11, 20, 1)]:
            self.__channels = int(np.abs(channels))
        else:
            raise ValueError(f'number of channels should be one of {[2**i for i in range(11, 20, 1)]}')

        self.real_time_unc = real_time_unc
        self.live_time_unc = live_time_unc
        
        if counts is None:
            self.counts = np.zeros(self.channels, dtype=int)
        else:
            self.counts = counts

        self.extended_dead_time = str(extended_dead_time)
        
        if chunk_info is not None:
            self.chunk_info = chunk_info
        else:
            self.chunk_info = []

        self.var = var
        self.datetime = None
        self.E_cal = None
        self.FWHM_cal = None
        self.o_filename = ''
        self.exp_params = {}
    
    @property
    def channels(self):
        """Set self.channels as a property
        """
        return self.__channels
        
    def __add__(self, other_chunk):
        """combine two spectra returning a new spectrum that is the sum of the two (sum of times and sum of counts)
        
        Parameters
        ----------
        other_chunk : GammaSpectrum (or SpectrumChunk)
            spectrum object
            
        Return
        ------
        sum_of_spectra : GammaSpectrum
            the object representing the sum of self and other_chunk
        """
        real_time = self.real_time + other_chunk.real_time
        live_time = self.live_time + other_chunk.live_time
        counts = self.counts + other_chunk.counts
        real_time_unc = np.sqrt(np.power(self.real_time_unc, 2) + np.power(other_chunk.real_time_unc, 2))
        live_time_unc = np.sqrt(np.power(self.live_time_unc, 2) + np.power(other_chunk.live_time_unc, 2))
        chunk_info_sum = self.chunk_info + other_chunk.chunk_info
        try:
            var = self.var + other_chunk.var
        except (TypeError, ValueError):
            var = None
        return GammaSpectrum(real_time, live_time, counts, real_time_unc, live_time_unc, len(counts), chunk_info_sum, var=var)
        
    def _get_chunks_info(self):
        """Aggregate spectrum information from every chunks
        
        Parameters
        ----------
        None
            
        Return
        ------
        spectrum_info : pandas.DataFrame
            DataFrame created from self.chunk_info list including relevant chunks information
        """
        return pd.concat(self.chunk_info, axis=1).T
        
    def defined_spectrum_integral(self, start=0, end=None):
        """Calculate the integral of a defined subset of the array counts
        
        Parameters
        ----------
        start : int
            start index to begin integration (default 0)
        end : int
            end index to integrate [excluded], if None default to len(self.counts) (default None)
            
        Return
        ------
        integral : int
            integral of the slice [start:end]
        """
        if end is None:
            end = len(self.counts)
        start, end = int(start), int(end)
        if start >= 0 and start < len(self.counts) and end > start and end <= len(self.counts):
            return np.sum(self.counts[start:end])
        else:
            return 0

    def input_count_rate(self):
        """Return the value for input count rate based on total number of recorded triggers and live time

        Parameters
        ----------
        None

        Return
        ------
        input_count_rate : float
            number of triggers / live time (unit s-1)
        input_count_rate_unc : float
            standard uncertainty of number of triggers / live time (unit s-1)
        """
        total_triggers = np.sum([item[('original_trigger', 'length')] for item in self.chunk_info])
        if self.live_time == 0 or total_triggers == 0:
            return 0.0, 0.0
        ICR = np.true_divide(total_triggers, self.live_time / 1E12)
        return ICR, ICR * np.sqrt(np.power(self.live_time_unc/self.live_time, 2) + np.power(1/np.sqrt(total_triggers), 2))

    def _save(self, filename=None, full=False):
        """save spectrum on disc
        
        Parameters
        ----------
        filename : str (or Path)
            filename of the saved spectrum (default None)
        full : bool
            descr
            
        Return
        ------
        None
        """
        today_time = datetime.datetime.today().strftime("%Y-%m-%d %H_%M_%S")
        if filename is None:
            asc_filename = f'spectrum_{today_time}_EDT_{self.extended_dead_time}.asc'
            csv_filename = f'spectrum_{today_time}_EDT_{self.extended_dead_time}.csv'
        else:
            asc_filename = f'{filename}_{today_time}_EDT_{self.extended_dead_time}.asc'
            csv_filename = f'{filename}_{today_time}_EDT_{self.extended_dead_time}.csv'
        start_time = self.datetime.strftime("%Y-%m-%dT%H:%M:%S") if self.datetime is not None else datetime.datetime(2000, 1, 1, 0, 0, 0).strftime("%Y-%m-%dT%H:%M:%S")
        with open(asc_filename, 'w') as wfile:
            for count in self.counts:
                wfile.write(f'{count}\n')
            wfile.write(f'#LiveTime={self.live_time / 1E12:.6f}\n')
            wfile.write(f'#LiveTime_unc={self.live_time_unc / 1E12:.6f}\n')
            wfile.write(f'#TrueTime={self.real_time / 1E12:.6f}\n')
            wfile.write(f'#TrueTime_unc={self.real_time_unc / 1E12:.6f}\n')
            wfile.write(f'#AcqStart={start_time}\n')
            wfile.write(f'#ExtDeadTime={self.extended_dead_time}\n')
        if full:
            df = self._get_chunks_info()
            df.to_csv(csv_filename)

    def _saven42(self, filename=None, add_info={}, full=False):
        """save spectrum on disc as N42 format

        Parameters
        ----------
        filename : str (or Path)
            filename of the saved spectrum (default None)
        add_info : dict
            dictioanry containig additional info not present in GammaSpectrum, such as start_date, E_coefficients
        full : bool
            dict (default False)

        Return
        ------
        None
        """

        def reconvert_time(timeint):
            timesec = timeint / 1E12

            #HOUR
            H = int(timesec // 3600)
            timesec = timesec % 3600
            if H > 0:
                H = f'{H}H'
            else:
                H = ''

            #MINUTE
            M = int(timesec // 60)
            timesec = timesec % 60
            if M > 0:
                M = f'{M}M'
            else:
                M = ''

            #SECOND
            S = f'{timesec:.6f}'

            return f'PT{H}{M}{S}S'

        with open('templ_n42.pkl', 'rb') as fl:
            tree = pickle.load(fl)

        root = tree.getroot()

        for item in root.iter('ChannelData'):
            item.text = ' '.join([f'{datum}' for datum in self.counts])

        for item in root.iter('StartDateTime'):
            if self.datetime is not None:
                dt = self.datetime
            else:
                dt = datetime.datetime(2000, 1, 1, 12, 0, 0)
            item.text = dt.isoformat()

        for item in root.iter('RealTimeDuration'):
            item.text = reconvert_time(self.real_time)

        for item in root.iter('LiveTimeDuration'):
            item.text = reconvert_time(self.live_time)

        for item in root.iter('CoefficientValues'):
            if self.E_cal is not None:
                cvs = self.E_cal
            else:
                cvs = np.array([0, 0, 0])
            item.text = ' '.join([f'{datum}' for datum in cvs])

        for item in root.iter('RadInstrumentManufacturerName'):
            item.text = 'created from list-mode via BIPM-INRIM script'

        tree.write("output.n42", encoding="utf-8", xml_declaration=True)

        if filename is None:
            today_time = datetime.datetime.today().strftime("%Y-%m-%d %H_%M_%S")
            n42_filename = f'spectrum_{today_time}_EDT_{self.extended_dead_time}.n42'
            csv_filename = f'spectrum_{today_time}_EDT_{self.extended_dead_time}.csv'
        elif filename.endswith('.n42'):
            n42_filename = filename
            csv_filename = f'{filename[:-4]}.csv'
        else:
            n42_filename = f'{filename}.n42'
            csv_filename = f'{filename}.csv'

        tree = ET.ElementTree(root)
        tree.write(n42_filename, encoding="utf-8", xml_declaration=True)

        if full:
            df = self._get_chunks_info()
            df.to_csv(csv_filename)

    def _preserve(self):
        preserve_name = os.path.splitext(self.o_filename)[0]

        with open(f'{preserve_name}.psp', 'wb') as pfile:
            pickle.dump(self, pfile)


class SpectrumChunkPHA(GammaSpectrum):#non-overlapping spectrum chunks, #consider also loss-free counting (LFC)
    """Class to manage non-overlapping chunks of information of the gamma spectrum (inherits from GammaSpectrum); optimized for list mode files obtained from PHA firmware

    Attributes
    ----------
    self.deadtimes : list
        list with timestamps of beginning and end of dead time
    self.start : int
        starting timestamp of the chunk
    self.end : int
        finishing timestamp of the chunk
    self.exclude : tuple
        tuple of strings indicating the codes to exclude while counting triggers

    Methods
    -------
    _count_counting_time()
        descr
    _count_counts()
        descr
    _extend_deadtime()
        descr
    _check_energy_overflow()
        descr
    _add_deadtime_to_trigger()
        descr
    """
    def __init__(self, trigger_times, extended_deadtime, pileuprejection_limit, _previous_endpoint, unc_trigger=1, channels=16384, LFC=False, LFC_type='ICR'):
        """SpectrumChunkPHA class constructor

        Parameters
        ----------
        trigger_times : list
            list of 3-items tuple for triggers (time, energy, flag)
        extended_deadtime : float
            extended dead time to apply to triggers, in microseconds
        pileuprejection_limit : float
            time limit in which two successive triggers are both excluded, in microseconds (default 0.5)
        _previous_endpoint : int
            timestamp indicating the endpoint of the previous chunk
        unc_trigger : float
            uncertainty of each timespan in microseconds (default 1)
        channels : int
            number of channels of spectrum (default 16384)
        LFC : bool
            whether perform elaboration in loss-free counts method, useful for spectra with rapidly changing dead times (default False)
        LFC_type : str
            kind of correction to perform to obtain 0 dead time (default 'ICR').
            valid options are:
            'ICR'  : correction based on the input count rate (1 - t * ICR)
            'time' : correction based on ratio between real and live times (real / live)

        Return
        ------
        None
        """
        super().__init__(channels=channels)
        #conversion of times from micro to picoseconds
        extended_deadtime = int(extended_deadtime * 1E6)
        pileuprejection_limit = int(pileuprejection_limit * 1E6)
        if pileuprejection_limit > extended_deadtime:
            raise ValueError(f'pileuprejection_limit ({pileuprejection_limit/1E6:.1f} μs) needs to be smaller than extended_deadtime ({extended_deadtime/1E6:.1f} μs)')
        unc_trigger = int(unc_trigger * 1E6)

        #info silenced for the moment
        chunk_info = {}
        chunk_info[('original_trigger', 'length')] = len(trigger_times)

        _trigger_codes = pd.Series([row[2] for row in trigger_times], dtype=object)
        _trigger_codes = _trigger_codes.unique()

        chunk_info[('original_trigger', 'flags')] = _trigger_codes

        #explanation about this one
        self.exclude = ('StartExcl', '0x4080', 'EOverFlow', 'PileupRej')#

        self.start = _previous_endpoint
        self.deadtimes = self._add_deadtime_to_trigger(trigger_times, extended_deadtime)
        try:
            if self.start > self.deadtimes[0][0]:
                self.deadtimes[0] = (self.start, self.deadtimes[0][1], self.deadtimes[0][2], 'StartExcl')
        except IndexError:
            pass

        self._extend_deadtime(pileuprejection_limit)

        dttr = np.array([_row[1] - _row[0] for _row in self.deadtimes])
        if len(dttr) == 0:
            dttr_min, dttr_max, dttr_ave = np.nan, np.nan, np.nan
        else:
            dttr_min, dttr_max, dttr_ave = np.min(dttr), np.max(dttr), np.average(dttr)

        chunk_info[('elaborated_trigger', 'length')] = len(self.deadtimes)
        chunk_info[('elaborated_trigger', 'min')] = dttr_min / 1E6
        chunk_info[('elaborated_trigger', 'average')] = dttr_ave / 1E6
        chunk_info[('elaborated_trigger', 'max')] = dttr_max / 1E6

        if len(self.deadtimes) > 0:
            self.end = self.deadtimes[-1][1]
        else:
            self.end = self.start

        self.real_time, self.live_time, self.real_time_unc, self.live_time_unc = self._count_counting_time(unc=unc_trigger)
        self.counts = self._count_counts()

        #Loss-Free Count mode
        if LFC:
            self.var = self.counts.copy() #evaluate uncertainty here
            if LFC_type == 'ICR':
                #OCR = len(trigger_times) / (self.live_time / 1E12)
                OCR = len(trigger_times) / (self.real_time / 1E12)
                #CRcorr = 1 / (1 - (extended_deadtime/1E12)*OCR)
                CRcorr = np.exp(extended_deadtime/1E12*OCR)
                self.counts = self.counts * CRcorr # exceptionally allowing an array of floats for self.counts
            else:
                self.counts = self.counts * self.real_time / self.live_time
            self.live_time = self.real_time
            self.live_time_unc = np.sqrt(np.power(self.live_time_unc,2) + np.power(self.real_time_unc,2))

        chunk_info['total counts'] = len(self.deadtimes)
        chunk_info['start'] = self.start
        chunk_info['end'] = self.end
        chunk_info['real time'] = self.real_time/1E12
        chunk_info['real time unc'] = self.real_time_unc/1E12
        chunk_info['live time'] = self.live_time/1E12
        chunk_info['live time unc'] = self.live_time_unc/1E12
        chunk_info['LFC'] = LFC

        self.chunk_info.append(pd.Series(chunk_info))

    def _check_energy_overflow(self, row, dead_time):
        """add trailing extended dead time to each trigger in signal_sequence

        Parameters
        ----------
        signal_sequence : list
            list of 3-items tuple for triggers (time, energy, flag)
        dead_time : int
            extended dead time to apply to triggers, in picoseconds

        Return
        ------
        extended_dead_time_signal_sequence : list
            list of 4-items tuple for triggers (time, time + extended dead time, energy, flag)
        """
        if row[1] == self.channels - 1:
            return row[0], row[0] + dead_time, row[1], 'EOverFlow'
        return row[0], row[0] + dead_time, row[1], row[2]#[self._check_energy_overflow(row) for row in signal_sequence]

    def _add_deadtime_to_trigger(self, signal_sequence, dead_time):
        """add trailing extended dead time to each trigger in signal_sequence

        Parameters
        ----------
        signal_sequence : list
            list of 3-items tuple for triggers (time, energy, flag)
        dead_time : int
            extended dead time to apply to triggers, in picoseconds

        Return
        ------
        extended_dead_time_signal_sequence : list
            list of 4-items tuple for triggers (time, time + extended dead time, energy, flag) resulting from function self._check_energy_overflow()
        """
        return [self._check_energy_overflow(row, dead_time) for row in signal_sequence]

    def _extend_deadtime(self, pileuprejection_limit):
        """extend the dead time in case of overlapping triggers

        Parameters
        ----------
        pileuprejection_limit : int
            time span (in ps) when 2 consecutive triggers are considered pileup and both rejected

        Return
        ------
        None
        """
        check_condition = [first[1] > second[0] for first, second in zip(self.deadtimes[:-1], self.deadtimes[1:])] + [False]

        _idx = 0
        while True:
            try:
                if check_condition[_idx] == True:
                    #something is happening
                    _edx = _idx + 1
                    while check_condition[_edx] == True:
                        _edx += 1

                    #aggregation of consecutive pulses to extend the dead time
                    #check pileup rejection
                    if self.deadtimes[_idx+1][0] - self.deadtimes[_idx][0] < pileuprejection_limit:
                        _flag = 'PileupRej'
                    else:
                        _flag = self.deadtimes[_idx][3]
                    self.deadtimes[_idx] = (self.deadtimes[_idx][0], self.deadtimes[_edx][1], self.deadtimes[_idx][2], _flag)

                    #deletion
                    for _sweepidx in range(1, _edx - _idx + 1, 1):
                        self.deadtimes[_idx + _sweepidx] = None
                    _idx = _edx + 1

                else:
                    _idx += 1

            except IndexError:
                break

        #filter
        self.deadtimes = [row for row in self.deadtimes if row is not None]

    def _count_counting_time(self, unc=1):
        """sum all deadtime periods to get total times

        Parameters
        ----------
        unc : int
            standard uncertainty on the single timestamp, in picoseconds (default 1)

        Return
        ------
        real_time : int
            real time of counting for the current chunk, in picoseconds
        live_time : int
            live time of counting for the current chunk, in picoseconds
        real_time_unc : int
            real time standard uncertainty of counting for the current chunk, in picoseconds
        live_time_unc : int
            live time standard uncertainty of counting for the current chunk, in picoseconds
        """
        diffs = [item[1] - item[0] for item in self.deadtimes]
        single_uncertainty = np.sqrt(2) * unc

        dead_time = np.sum(diffs)
        dead_time_unc = np.sqrt(len(diffs)) * single_uncertainty

        real_time = self.end - self.start
        real_time_unc = single_uncertainty

        live_time = real_time - dead_time
        live_time_unc = np.sqrt(np.power(single_uncertainty, 2) + np.power(dead_time_unc, 2))
        return real_time, live_time, real_time_unc, live_time_unc

    def _count_counts(self):
        """convert the list of channels from the chunk to an ordered spectrum array

        Parameters
        ----------
        None

        Return
        ------
        counts : np.array
            array of int indicating number of counts for each ADC channel with index as the channel number
        """
        energies = np.array([item[2] for item in self.deadtimes if item[3] not in self.exclude], dtype=int)
        counts = pd.Series(energies).value_counts(sort=False, ascending=True)
        return np.array([counts.get(key, 0) for key in range(self.channels)], dtype=int)


class SpectrumChunk2channelPHA(GammaSpectrum):#non-overlapping spectrum chunks, #consider also loss-free counting (LFC)
    """Class to manage non-overlapping chunks of information of the gamma spectrum (inherits from GammaSpectrum); optimized for list mode files obtained from PHA firmware

    Attributes
    ----------
    self.deadtimes : list
        list with timestamps of beginning and end of dead time
    self.start : int
        starting timestamp of the chunk
    self.end : int
        finishing timestamp of the chunk
    self.exclude : tuple
        tuple of strings indicating the codes to exclude while counting triggers

    Methods
    -------
    _count_counting_time()
        descr
    _count_counts()
        descr
    _extend_deadtime()
        descr
    _check_energy_overflow()
        descr
    _add_deadtime_to_trigger()
        descr
    """
    def __init__(self, trigger_times, gate_times, extended_deadtime, pileuprejection_limit, _previous_endpoint, unc_trigger=1, channels=16384, LFC=False, LFC_type='ICR'):
        """SpectrumChunkPHA class constructor

        Parameters
        ----------
        trigger_times : list
            list of 3-items tuple for triggers (time, energy, flag)
        extended_deadtime : float
            extended dead time to apply to triggers, in microseconds
        pileuprejection_limit : float
            time limit in which two successive triggers are both excluded, in microseconds (default 0.5)
        _previous_endpoint : int
            timestamp indicating the endpoint of the previous chunk
        unc_trigger : float
            uncertainty of each timespan in microseconds (default 1)
        channels : int
            number of channels of spectrum (default 16384)
        LFC : bool
            whether perform elaboration in loss-free counts method, useful for spectra with rapidly changing dead times (default False)
        LFC_type : str
            kind of correction to perform to obtain 0 dead time (default 'ICR').
            valid options are:
            'ICR'  : correction based on the input count rate (1 - t * ICR)
            'time' : correction based on ratio between real and live times (real / live)

        Return
        ------
        None
        """
        super().__init__(channels=channels)
        #conversion of times from micro to picoseconds
        extended_deadtime = int(extended_deadtime * 1E6)
        pileuprejection_limit = int(pileuprejection_limit * 1E6)
        if pileuprejection_limit > extended_deadtime:
            raise ValueError(f'pileuprejection_limit ({pileuprejection_limit/1E6:.1f} μs) needs to be smaller than extended_deadtime ({extended_deadtime/1E6:.1f} μs)')
        unc_trigger = int(unc_trigger * 1E6)

        #info silenced for the moment
        chunk_info = {}
        #chunk_info[('original_trigger', 'length')] = len(trigger_times)

        _trigger_codes = pd.Series([row[2] for row in trigger_times], dtype=object)
        _trigger_codes = _trigger_codes.unique()

        #chunk_info[('original_trigger', 'flags')] = _trigger_codes

        #explanation about this one
        self.exclude = ('StartExcl', '0x4080', 'EOverFlow', 'PileupRej')#

        self.start = _previous_endpoint
        self.deadtimes = self._add_deadtime_to_trigger(trigger_times, extended_deadtime)
        try:
            if self.start > self.deadtimes[0][0]:
                self.deadtimes[0] = (self.start, self.deadtimes[0][1], self.deadtimes[0][2], 'StartExcl')
        except IndexError:
            pass

        print('GATE TIMES')
        print(gate_times)
        print('DEAD TIMES')
        print(self.deadtimes)

        #self._extend_deadtime(pileuprejection_limit)

        #dttr = np.array([_row[1] - _row[0] for _row in self.deadtimes])
        #if len(dttr) == 0:
        #    dttr_min, dttr_max, dttr_ave = np.nan, np.nan, np.nan
        #else:
        #    dttr_min, dttr_max, dttr_ave = np.min(dttr), np.max(dttr), np.average(dttr)

        #chunk_info[('elaborated_trigger', 'length')] = len(self.deadtimes)
        #chunk_info[('elaborated_trigger', 'min')] = dttr_min / 1E6
        #chunk_info[('elaborated_trigger', 'average')] = dttr_ave / 1E6
        #chunk_info[('elaborated_trigger', 'max')] = dttr_max / 1E6

        if len(self.deadtimes) > 0:
            self.end = self.deadtimes[-1][1]
        else:
            self.end = self.start

        self.real_time, self.live_time, self.real_time_unc, self.live_time_unc = self._count_counting_time(unc=unc_trigger)
        self.counts = self._count_counts()

    def _explore(self, gate_times):
        gt_idx = 0
        for row in self.deadtimes:
            print(row)
            while True:
                try:
                    if gate_times[gt_idx][0] < row[0] and gate_times[gt_idx+1][0] > row[0]:
                        break
                except IndexError:
                    pass
                gt_idx += 1

    def _check_energy_overflow(self, row, dead_time):
        """add trailing extended dead time to each trigger in signal_sequence

        Parameters
        ----------
        signal_sequence : list
            list of 3-items tuple for triggers (time, energy, flag)
        dead_time : int
            extended dead time to apply to triggers, in femtoseconds

        Return
        ------
        extended_dead_time_signal_sequence : list
            list of 4-items tuple for triggers (time, time + extended dead time, energy, flag)
        """
        if row[1] == self.channels - 1:
            return row[0], row[0] + dead_time, row[1], 'EOverFlow'
        return row[0], row[0] + dead_time, row[1], row[2]#[self._check_energy_overflow(row) for row in signal_sequence]

    def _add_deadtime_to_trigger(self, signal_sequence, dead_time):
        """add trailing extended dead time to each trigger in signal_sequence

        Parameters
        ----------
        signal_sequence : list
            list of 3-items tuple for triggers (time, energy, flag)
        dead_time : int
            extended dead time to apply to triggers, in femtoseconds

        Return
        ------
        extended_dead_time_signal_sequence : list
            list of 4-items tuple for triggers (time, time + extended dead time, energy, flag) resulting from function self._check_energy_overflow()
        """
        return [self._check_energy_overflow(row, dead_time) for row in signal_sequence]

    def _count_counting_time(self, unc=1):
        """sum all deadtime periods to get total times

        Parameters
        ----------
        unc : int
            standard uncertainty on the single timestamp, in picoseconds (default 1)

        Return
        ------
        real_time : int
            real time of counting for the current chunk, in picoseconds
        live_time : int
            live time of counting for the current chunk, in picoseconds
        real_time_unc : int
            real time standard uncertainty of counting for the current chunk, in picoseconds
        live_time_unc : int
            live time standard uncertainty of counting for the current chunk, in picoseconds
        """
        diffs = [item[1] - item[0] for item in self.deadtimes]
        single_uncertainty = np.sqrt(2) * unc

        dead_time = np.sum(diffs)
        dead_time_unc = np.sqrt(len(diffs)) * single_uncertainty

        real_time = self.end - self.start
        real_time_unc = single_uncertainty

        live_time = real_time - dead_time
        live_time_unc = np.sqrt(np.power(single_uncertainty, 2) + np.power(dead_time_unc, 2))
        return real_time, live_time, real_time_unc, live_time_unc

    def _count_counts(self):
        """convert the list of channels from the chunk to an ordered spectrum array

        Parameters
        ----------
        None

        Return
        ------
        counts : np.array
            array of int indicating number of counts for each ADC channel with index as the channel number
        """
        energies = np.array([item[2] for item in self.deadtimes if item[3] not in self.exclude], dtype=int)
        counts = pd.Series(energies).value_counts(sort=False, ascending=True)
        return np.array([counts.get(key, 0) for key in range(self.channels)], dtype=int)


class SpectrumChunkFPGA(GammaSpectrum):#non-overlapping spectrum chunks, #consider also loss-free counting (LFC)
    """Class to manage non-overlapping chunks of information of the gamma spectrum (inherits from GammaSpectrum); optimized for list mode files obtained from PHA firmware

    Attributes
    ----------
    self.deadtimes : list
        list with timestamps of beginning and end of dead time
    self.start : int
        starting timestamp of the chunk
    self.end : int
        finishing timestamp of the chunk
    self.exclude : tuple
        tuple of strings indicating the codes to exclude while counting triggers

    Methods
    -------
    _count_counting_time()
        descr
    _count_counts()
        descr
    _extend_deadtime()
        descr
    _check_energy_overflow()
        descr
    _add_deadtime_to_trigger()
        descr
    """
    def __init__(self, trigger_dataframe, _previous_endpoint, extended_deadtime, ADC_conv=256, unc_trigger=1, channels=16384, LFC=False, LFC_type='ICR'):
        """SpectrumChunkPHA class constructor

        Parameters
        ----------
        trigger_dataframe : list
            list of 3-items tuple for triggers (time, energy, flag)
        _previous_endpoint : int
            timestamp indicating the endpoint of the previous chunk
        extended_deadtime : float
            extended dead time to apply to triggers, in microseconds
        unc_trigger : float
            uncertainty of each timespan in microseconds (default 1)
        channels : int
            number of channels of spectrum (default 16384)
        LFC : bool
            whether perform elaboration in loss-free counts method, useful for spectra with rapidly changing dead times (default False)
        LFC_type : str
            kind of correction to perform to obtain 0 dead time (default 'ICR').
            valid options are:
            'ICR'  : correction based on the input count rate (1 - t * ICR)
            'time' : correction based on ratio between real and live times (real / live)

        Return
        ------
        None
        """
        super().__init__(channels=channels)
        #conversion of times from micro to picoseconds

        extended_deadtime = int(extended_deadtime * 1E6)
        self.start = _previous_endpoint
        self.deadtimes = trigger_dataframe
        self.exclude = trigger_dataframe['CHARGE'] < 0

        chunk_info = {}
        chunk_info[('original_trigger', 'length')] = len(self.deadtimes)

        self.real_time, self.live_time, self.real_time_unc, self.live_time_unc = self._count_counting_time(unc=unc_trigger)
        self.counts = self._count_counts(ADC_conv)

        if len(self.deadtimes) > 0:
            self.end = self.start + self.real_time
        else:
            self.end = self.start

        #Loss-Free Count mode
        if LFC:
            self.var = self.counts.copy() #evaluate uncertainty here
            if LFC_type == 'ICR':
                OCR = len(trigger_times) / (self.real_time / 1E12)
                CRcorr = np.exp(extended_deadtime/1E12*OCR)
                self.counts = self.counts * CRcorr
            else:
                self.counts = self.counts * self.real_time / self.live_time
            self.live_time = self.real_time
            self.live_time_unc = np.sqrt(np.power(self.live_time_unc,2) + np.power(self.real_time_unc,2))

        chunk_info['total counts'] = len(self.deadtimes[~self.exclude])
        chunk_info['start'] = self.start
        chunk_info['end'] = self.end
        chunk_info['real time'] = self.real_time/1E12
        chunk_info['real time unc'] = self.real_time_unc/1E12
        chunk_info['live time'] = self.live_time/1E12
        chunk_info['live time unc'] = self.live_time_unc/1E12
        chunk_info['LFC'] = LFC

        self.chunk_info.append(pd.Series(chunk_info))


    def _count_counting_time(self, unc=1, sample_to_picos_conv=8000):
        """sum all deadtime periods to get total times

        Parameters
        ----------
        unc : int
            standard uncertainty on the single timestamp, in picoseconds (default 1)

        Return
        ------
        real_time : int
            real time of counting for the current chunk, in picoseconds
        live_time : int
            live time of counting for the current chunk, in picoseconds
        real_time_unc : int
            real time standard uncertainty of counting for the current chunk, in picoseconds
        live_time_unc : int
            live time standard uncertainty of counting for the current chunk, in picoseconds
        """
        N = len(self.deadtimes)
        single_uncertainty = np.sqrt(2) * unc

        dead_time = np.sum(self.deadtimes['BUSY']) * sample_to_picos_conv
        dead_time_unc = np.sqrt(N) * single_uncertainty * sample_to_picos_conv

        live_time = np.sum(self.deadtimes['IDLE']) * sample_to_picos_conv
        live_time_unc = dead_time_unc

        real_time = live_time + dead_time
        real_time_unc = np.sqrt(2) * live_time_unc

        return real_time, live_time, real_time_unc, live_time_unc

    def _count_counts(self, ADC_conv):
        """convert the list of channels from the chunk to an ordered spectrum array

        Parameters
        ----------
        None

        Return
        ------
        counts : np.array
            array of int indicating number of counts for each ADC channel with index as the channel number
        """
        energies = self.deadtimes['CHARGE'][~self.exclude]/ADC_conv
        energies = energies.astype(int)
        counts = energies.value_counts(sort=False, ascending=True)
        return np.array([counts.get(key, 0) for key in range(self.channels)], dtype=int)


def openASCfile(file):
    """Open an asc file and retrieve spectrum information
    
    Parameters
    ----------
    file : str (path)
        name of the list mode file
    
    Return
    ------
    real : float
        acquisition real time in s
    live : float
        acquisition live time in s
    spectrum_counts : np.array
        array of int indicating counts for each channel
    real_unc : float
        uncertainty of real time in s
    live_unc : float
        uncertianty of live time in s
    startcounting : datetime
        date of start acquisition
    ext_deadtime : str
        extended dead time applied
    """
    with open(file,'r') as ascfile:
        filelines = [line.replace('\n','') for line in ascfile.readlines()]
    #optionals
    ext_deadtime = '-1'
    
    idx = 0
    for line in filelines[::-1]:
        if '#ExtDeadTime=' in line:
            ext_deadtime = line.replace(f'#ExtDeadTime=','')
        if '#AcqStart=' in line:
            date_time = line.replace('#AcqStart=','')
        if '#TrueTime=' in line:
            real = float(line.replace('#TrueTime=',''))
        if '#TrueTime_unc=' in line:
            real_unc = float(line.replace('#TrueTime_unc=',''))
        if '#LiveTime_unc=' in line:
            live_unc = float(line.replace('#LiveTime_unc=',''))
        if '#LiveTime=' in line:
            live = float(line.replace('#LiveTime=',''))
            idx = filelines.index(line)
            break
    startcounting = datetime.datetime.strptime(date_time, "%Y-%m-%dT%H:%M:%S")
    spectrum_counts = np.array([int(iks) for iks in filelines[:idx]])
    return real, live, spectrum_counts, real_unc, live_unc, startcounting, ext_deadtime

def open_n42_file(file):
    """Open a spectrum file formatted as ANSI/IEEE N42.42 and retrieve spectrum information

    Parameters
    ----------
    file : str (path)
        name of the N42 file

    Return
    ------
    real : float
        acquisition real time in s
    live : float
        acquisition live time in s
    spectrum_counts : np.array
        array of int indicating counts for each channel
    real_unc : float
        uncertianty of real time in s
    live_unc : float
        uncertianty of live time in s
    startcounting : datetime
        date of start acquisition
    ext_deadtime : str
        extended dead time applied
    """
    def convert_time(timestring):
        cleaned = timestring.lstrip('PT')
        #HOUR
        if 'H' in cleaned:
            H, cleaned = cleaned.split('H')
            H = int(H)
        else:
            H = 0

        #MINUTE
        if 'M' in cleaned:
            M, cleaned = cleaned.split('M')
            M = int(M)
        else:
            M = 0

        #SECOND
        S = float(cleaned.replace('S', ''))

        return H * 3600 + M * 60 + S

    tree = ET.parse(file)
    root = tree.getroot()

    for item in root.iter('ChannelData'):
        spectrum_counts = np.array([int(datum) for datum in item.text.split()], dtype=int)

    for item in root.iter('LiveTimeDuration'):
        live = convert_time(item.text)

    for item in root.iter('RealTimeDuration'):
        real = convert_time(item.text)

    for item in root.iter('StartDateTime'):
        startcounting = datetime.datetime.fromisoformat(item.text)

    for item in root.iter('CoefficientValues'):
        cvs = np.array([float(coeff) for coeff in item.text.split()])

    return real, live, spectrum_counts, 0.02, 0.02, startcounting, '-1'

def open_events_file(path, columns=('BOARD', 'CHANNEL', 'TIMETAG', 'ENERGY', 'FLAGS', 'PROBE_CODE')):
    """open csv list mode spectrum file and returns a pandas dataframe

    Parameters
    ----------
    path : str (path)
        name of the gamma spectrum in list mode file
    columns : tuple
        name of the columns for the resulting pd.DataFrame (default ('BOARD', 'CHANNEL', 'TIMETAG', 'ENERGY', 'FLAGS', 'PROBE_CODE'))

    Return
    ------
    events_data : pd.DataFrame
        Series containing the most relevant information about the spectrum
    """
    return pd.read_csv(path, sep=';', header=None, skiprows=1, names=columns, usecols=[_ for _ in range(len(columns))], comment='#')

def open_preserved_object(pickled_filename):
    """general function to recall serialized objects (usually identified as the first letter of their extension is a p such as '.psp' or '.pel')

    Parameters
    ----------
    pickled_filename : str (path)
        name of the gamma spectrum in list mode file

    Return
    ------
    object : object
        original serialized object
    """
    with open(pickled_filename, 'rb') as handle:
        return pickle.load(handle)

#deprecated : not relevant anymore since the function on top (open_preserved_object) is more general
#def open_preserved_spectrum(pickled_spectrum_filename):
#    with open(pickled_spectrum_filename, 'rb') as handle:
#        return pickle.load(handle)

def _make_spectrum(spectrum, spectrum_counter, reference_date, Nfit, uNfit, Nint, uNint, pupcorr, upupcorr, centr=0, source=0, spectrum_collection=None):
    """Save current spectrum and return newly initialized spectrum and updated spectrum_counter and reference_date

    Parameters
    ----------
    spectrum : str (path)
        name of the gamma spectrum in list mode file
    spectrum_counter : int
        counter of the current spectrum
    reference_date : int
        descr
    Nfit : float
        net area evaluated by fitting procedure
    uNfit : float
        uncertainty of Nfit
    Nint : float
        net area evaluated by numeric integration of peak
    uNint : float
        uncertainty of Nint
    pupcorr : float
        value for pileup correction factor
    upupcorr : float
        uncertainty of pileup correcion
    centr : float
        fitted centroid of the investigated peak (default 0)
    source : int
        descr (default 0)
    spectrum_collection : list (or None)
        list to store the created spectrum objects (default None)

    Return
    ------
    spectrum : GammaSpectrum
        newly initialized spectrum
    spectrum_counter : int
        spectrum_counter + 1
    reference_date : int
        reference_date + real time of the previous spectrum
    s_info : pd.Series
        Series containing the most relevant information about the spectrum
    """
    spectrum_channels = spectrum.channels
    extended_deadtime = spectrum.extended_dead_time

    spectrum.datetime = reference_date

    add_info = {'start_date':reference_date}
    fname=f'spectrum_{source}_EDT{extended_deadtime}_{spectrum_counter:03d}.n42'
    spectrum.o_filename = fname
    spectrum._saven42(filename=fname, add_info=add_info, full=False)

    if spectrum_collection is not None:
        spectrum_collection.append(spectrum)

    #important data for decay measurement
    data = {'id':spectrum_counter, 'meas_date':reference_date, 'meas_time':spectrum.real_time/1E12, 'live_time':spectrum.live_time/1E12, 'live_time_u':spectrum.live_time_unc/1E12, 'meas_dead_time':1-np.true_divide(spectrum.live_time, spectrum.real_time), 'ext_dead_time':extended_deadtime, 'N_fit':Nfit, 'uN_fit':uNfit, 'N_int':Nint, 'uN_int':uNint, 'pupcorr':pupcorr, 'upupcorr':upupcorr, 'centroid':centr, 'filename':fname}
    s_info = pd.Series(data)

    #initialize new spectrum
    reference_date += datetime.timedelta(microseconds=spectrum.real_time/1E6)
    spectrum = GammaSpectrum(channels=spectrum_channels, var=np.zeros(spectrum_channels))
    spectrum.extended_dead_time = str(extended_deadtime)
    spectrum_counter += 1

    return spectrum, spectrum_counter, reference_date, s_info, spectrum_collection

def _produce_decay_dataframe(decay_data, source=0):
    """Gather and combine information concerning decay spectra into a pandas.DataFrame for half-life evaluation

    Parameters
    ----------
    decay_data : list (of pd.Series)
        list of multiple pd.Series to concatenate in a single pandas.DataFrame
    source : int
        index to identify the decay (default 0)

    Return
    ------
    decay_data : pd.DataFrame
        newly initialized spectrum
    """
    decay_data = pd.concat(decay_data, axis=1).T

    decay_data['id'] = decay_data['id'].astype(int)
    decay_data['meas_time'] = decay_data['meas_time'].astype(float)
    decay_data['live_time'] = decay_data['live_time'].astype(float)
    decay_data['live_time_u'] = decay_data['live_time_u'].astype(float)
    decay_data['meas_dead_time'] = decay_data['meas_dead_time'].astype(float)
    decay_data['N_fit'] = decay_data['N_fit'].astype(float)
    decay_data['uN_fit'] = decay_data['uN_fit'].astype(float)
    decay_data['N_int'] = decay_data['N_int'].astype(float)
    decay_data['uN_int'] = decay_data['uN_int'].astype(float)
    decay_data['pupcorr'] = decay_data['pupcorr'].astype(float)
    decay_data['upupcorr'] = decay_data['upupcorr'].astype(float)
    decay_data['meas_date'] = pd.to_datetime(decay_data['meas_date'])

    decay_data['Act'] = decay_data['N_fit'] / decay_data['live_time'] * decay_data['pupcorr']
    decay_data['Act_u'] = decay_data['Act'] * np.sqrt(np.power(decay_data['uN_fit']/decay_data['N_fit'], 2) + np.power(decay_data['live_time_u']/decay_data['live_time'], 2) + np.power(decay_data['upupcorr']/decay_data['pupcorr'], 2))

    decay_data['coincidence_w'] = 'X'
    decay_data['cocktail'] = 'X'
    decay_data['birks'] = 'X'
    decay_data['source'] = str(source)

    return decay_data

def safelim(data, default, sign='+'):
    """Return visualization vertical limit for residuals online plot

    Parameters
    ----------
    default : numeric
        filename of the list mode file
    sign : str
        sign of the limit, it is different while considering positive values or negative ones [options "-", "+"] (default "+")

    Return
    ------
    value : numeric
        value of the limit
    """
    if sign == '+':
        value = np.max([np.max(np.abs(data))+0.1, default])
        if np.isinf(value) or np.isnan(value):
            return default
        return value
    value = np.min([-np.max(np.abs(data))-0.1, -default])
    if np.isinf(value) or np.isnan(value):
        return -default
    return value

def retrieve_date(filename):
    """Return datetime object corresponding to the date of start acquisition by reading the 2nd line of the list mode file

    Parameters
    ----------
    filename : str (or path)
        filename of the list mode file

    Return
    ------
    start_datetime : datetime.datetime (or None)
        datetime object, time of start acquisition
    """
    with open(filename, 'r') as header:
        header.readline()
        isodate = header.readline().strip()
    if isodate[0] == '#':
        try:
            return datetime.fromisoformat(isodate[1:])
        except Exception:
            return None
    return None

def retrieve_Ecalibration(filename):
    """Return array of energy calibration parameters by reading the 3rd line of the list mode file

    Parameters
    ----------
    filename : str (or path)
        filename of the list mode file

    Return
    ------
    calibration_parameters : np.array
        array of parameters (with increasing power) describing the energy calibration
    """
    with open(filename, 'r') as header:
        header.readline()
        header.readline()
        calib_params = header.readline().strip()
    if calib_params[0] == '#':
        try:
            return np.array([float(item) for item in calib_params[1:].split()], dtype=float)
        except Exception:
            return np.array([0, 1, 0], dtype=float)
    return np.array([0, 1, 0], dtype=float)

#filename, chunksize=100000, channel_signal=None, extended_deadtime=100, pileuprejection_limit=0.5, spectrum_channels=16384, verbose=True, LFC=False, LFC_type='ICR'
def open_digitized_2channelspectrumPHA(filename_triggers, filename_gate, chunksize=100000, extended_deadtime=100, pileuprejection_limit=0.5, spectrum_channels=16384, verbose=True, LFC=False, LFC_type='ICR'):

    date_from_trigger_file = retrieve_date(filename_triggers)
    date_from_gate_file = retrieve_date(filename_gate)

    if date_from_trigger_file != date_from_gate_file:
        raise TypeError('The two list mode files need to have the same start acquisition date')

    Ecalib_fromfile = retrieve_Ecalibration(filename_triggers)

    with pd.read_csv(filename_triggers, sep=';', header=0, names=('BOARD', 'CHANNEL', 'TIMETAG', 'ENERGY', 'FLAGS', 'PROBE_CODE'), chunksize=chunksize, comment='#') as spectrum_file:
        #initialize the spectrum
        spectrum = GammaSpectrum(channels=spectrum_channels, var=np.zeros(spectrum_channels))
        spectrum.extended_dead_time = str(extended_deadtime)

        _previous_endpoint = 0

        for chunk in spectrum_file:

            minTT = np.min(chunk['TIMETAG'])
            maxTT = np.max(chunk['TIMETAG'])

            gate_chunk = []
            with pd.read_csv(filename_gate, sep=';', header=0, names=('BOARD', 'CHANNEL', 'TIMETAG', 'ENERGY', 'FLAGS', 'PROBE_CODE'), chunksize=chunksize, comment='#') as gate_file:
                for chunkG in gate_file:
                    if np.max(chunkG['TIMETAG']) < minTT or np.min(chunkG['TIMETAG']) > maxTT:
                        pass
                    else:
                        gate_chunk.append(chunkG)
            gate_chunk = pd.concat(gate_chunk)
            _Tfilt = (gate_chunk['TIMETAG'] > minTT) & (gate_chunk['TIMETAG'] < maxTT)

            local_ch = SpectrumChunk2channelPHA([(trigger_time, energy, flag) for trigger_time, energy, flag in zip(chunk['TIMETAG'], chunk['ENERGY'], chunk['PROBE_CODE'])], [(trigger_time, flag) for trigger_time, flag in zip(gate_chunk[_Tfilt]['TIMETAG'], gate_chunk[_Tfilt]['PROBE_CODE'])], extended_deadtime, pileuprejection_limit, _previous_endpoint, channels=spectrum.channels, LFC=LFC, LFC_type=LFC_type)

            _previous_endpoint = local_ch.end
            #add chunk's information to the total spectrum
            spectrum += local_ch
            if verbose:
                if LFC:
                    comment = 'LFC'
                else:
                    comment = ''
                print(f'REAL: {spectrum.real_time / 1E12:.1f} s ({(1-spectrum.live_time/spectrum.real_time)*100:.1f} %) {comment}')

    spectrum.extended_dead_time = str(extended_deadtime)
    if LFC:
        spectrum.counts = np.round(spectrum.counts).astype(int)

    spectrum.datetime = date_from_trigger_file if date_from_trigger_file is not None else datetime.datetime(2000, 1, 1, 12, 0, 0)
    spectrum.E_cal = Ecalib_fromfile

    spectrum.o_filename = filename_triggers

    return spectrum



def manage_stream_digitized_data(filename, chunksize=100000, channel_signal=None, extended_deadtime=100, pileuprejection_limit=0.5, spectrum_channels=16384, verbose=True, LFC=False, LFC_type='ICR', condition_options={}, peak_animation=True, source=0, exclude_last=False, hard_time_limit=False, two_tau=0.0, two_tau_unc=0.0, fit=''):#maybe loss-free counting (LFC)
    """Save a series of spectra from the stream of a single list mode file and return a summary dataframe

    Parameters
    ----------
    filename : str (path)
        name of the gamma spectrum in list mode file
    chunksize : int
        number of rows to elaborate at each iteration; corresponds to the resolution of the peak since conditions are checked after each chunk (default 100000)
    channel_signal : int
        descr (default None)
    extended_deadtime : float
        time in us (default 100)
    pileuprejection_limit : float
        time limit in which two successive triggers are both excluded, in microseconds (default 0.5)
    spectrum_channels : int
        number of channels in the spectrum (default 16384)
    verbose : bool
        display information during elaboration (default True)
    LFC : bool
        whether perform elaboration in loss-free counts method, useful for spectra with rapidly changing dead times (default False)
    LFC_type : str
        kind of correction to perform to obtain 0 dead time (default 'ICR').
        valid options are:
        'ICR'  : correction based on the input count rate (1 - t * ICR)
        'time' : correction based on ratio between real and live times (real / live)
    condition_options : dict
        dict with conditions to trigger the closure of a spectrum. Can contain pairs key : value, such as
        'real_time' : numeric #real time in seconds
        'live_time' : numeric #live time in seconds
        'centroid'  : int #centroid of the peak (int)
        'peak_runc' : float #percent relative uncertainty of peak (float)
        'roi'       : (tuple of ints of lenght 2) #limits of the ROI as int indexes (lower-limit, upper-limit)
        'bkg_nch'   : (tuple of ints of lenght 2) #number of channles to use for background calculation (lower-background channels, upper-background channels)
    peak_animation : bool
        whether to show the peak fitting in the selected ROI (default True)
    source : int
        id for the stream data, a nice way to discriminate between multiple decays for the same isotope (default 0)
    exclude_last : bool
        whether to exclude the last chunk of incomplete information (default False)
    hard_time_limit : bool
        whether to set the real time parameter as a hard limit and stop the elaboration if it's reached (default False)
    two_tau : float
        2t experimental value in microseconds defined as the gradient of lnA versus R (default 0.0)
    two_tau_unc : float
        standard uncertainty of two_tau in microseconds (default 0.0)
    fit : str
        code to recall a specific fit from peaks module (default "")

    Return
    ------
    decay_data_df : pd.DataFrame
        dataframe containing relevant information of all the saved spectra
    spectra_collection : list
        list of spectrum objects obtained from the elaboration
    """

    cond_RT = condition_options.get('real_time', None)
    cond_LT = condition_options.get('live_time', None)
    cond_PK = condition_options.get('centroid', None)
    cond_PK_UNC = condition_options.get('peak_runc', None)
    cond_ROI = condition_options.get('roi', None)
    cond_BKG_CH = condition_options.get('bkg_nch', (10, 10))
    decay_data = []
    spectra_collection = []

    if cond_RT is None and cond_LT is None and cond_PK_UNC is None:
        raise ValueError('at least one condition of "real_time", "live_time" and "centroid" should be selected')

    #fit selection
    _peakfit, _BKGfit = pk_elab._get_fit(fit)

    #date information
    date_from_file = retrieve_date(filename)
    E_calib = retrieve_Ecalibration(filename)

    #for pileup correction factor
    try:
        two_tau, two_tau_unc = float(two_tau)/1E6, float(two_tau_unc)/1E6
    except (TypeError, ValueError):
        two_tau, two_tau_unc = 0.0, 0.0

    with pd.read_csv(filename, sep=';', header=0, names=('BOARD', 'CHANNEL', 'TIMETAG', 'ENERGY', 'FLAGS', 'PROBE_CODE'), chunksize=chunksize, comment='#') as spectrum_file:
        #initialize the spectrum
        spectrum = GammaSpectrum(channels=spectrum_channels, var=np.zeros(spectrum_channels))
        spectrum.extended_dead_time = str(extended_deadtime)

        _previous_endpoint = 0
        reference_date = date_from_file if date_from_file is not None else datetime.datetime.today()
        spectrum_counter = 0

        #default fit parameters
        X = np.arange(cond_ROI[0], cond_ROI[1]+1)
        if peak_animation:
            plt.ion()
            fig, (ax, axres) = plt.subplots(nrows=2, ncols=1, sharex=True, height_ratios=[2.5, 1])
            yinit = np.array([np.nan for _ in X])
            _line_spectrum = ax.plot(X, yinit, linestyle='-', color='grey', linewidth=1, marker='')
            _line_fit = ax.plot(X, yinit, linestyle='-', color='r', linewidth=0.75, marker='', alpha=0.7)
            _line_background = ax.plot(X, yinit, linestyle='-', color='#006400', linewidth=0.75, marker='', alpha=0.7)
            _title = ax.set_title(f"{spectrum_counter:03d} | real: {spectrum.real_time / 1E12:.0f} s")
            ax.set_xlim(np.min(X), np.max(X))

            _res_fit = axres.plot(X, yinit, linestyle='-', color='r', linewidth=0.75, marker='', alpha=0.7)
            axres.axhline(y=0, color='k', linestyle='-', linewidth=0.75)
            axres.axhline(y=-3, color='#306AC0', linestyle='-', linewidth=1.5)
            axres.axhline(y=3, color='#306AC0', linestyle='-', linewidth=1.5)
            axres.set_ylim(-3, 3)

        for chunk in spectrum_file:

            #no need to filter the signal by channel again, but just to be sure
            if isinstance(channel_signal, int):
                _filter_signal = chunk['CHANNEL'] == channel_signal
            else:
                _filter_signal = np.ones(len(chunk['CHANNEL']), dtype=bool)

            local_ch = SpectrumChunkPHA([(trigger_time, energy, flag) for trigger_time, energy, flag in zip(chunk[_filter_signal]['TIMETAG'], chunk[_filter_signal]['ENERGY'], chunk[_filter_signal]['PROBE_CODE'])], extended_deadtime, pileuprejection_limit, _previous_endpoint, channels=spectrum.channels, LFC=LFC, LFC_type=LFC_type)

            _previous_endpoint = local_ch.end
            #add chunk's information to the total spectrum
            spectrum += local_ch
            if verbose:
                if LFC:
                    comment = 'LFC'
                else:
                    comment = ''
                print(f'REAL: {spectrum.real_time / 1E12:.1f} s ({(1-spectrum.live_time/spectrum.real_time)*100:.1f} %) {comment}')

            fit_peak, urfit_peak = 0.0, 1.0
            int_peak, urint_peak = 0.0, 1.0
            pup_corr, upup_corr = 1.0, 0.0
            centr = 0

            #check all conditions anyway?
            if cond_ROI[1] > cond_PK and cond_ROI[0] < cond_PK:
                Y = spectrum.counts[X] + 0.001
                try:
                    Y2 = spectrum.var[X] + 0.001
                except Exception:
                    Y2 = None
                popt, pcov = pk_elab._manage_fit(X, Y, cond_PK, _peakfit, Y2=Y2)
                if popt[0] is not None:
                    fit_peak = popt[0]
                    urfit_peak = (np.sqrt(np.diag(pcov))[0] / fit_peak) * 100
                else:
                    fit_peak = 0
                    urfit_peak = 100

                spcICR, uspcICR = spectrum.input_count_rate()
                pup_corr, upup_corr = pk_elab.evaluate_pileup_correction(spcICR, uspcICR, two_tau, two_tau_unc)
                centr = popt[1]

            if cond_BKG_CH[0] + cond_BKG_CH[1] < cond_ROI[1] - cond_ROI[0]:
                int_peak, urint_peak = pk_elab._peak_integral(spectrum, cond_ROI, cond_BKG_CH)

            if peak_animation:

                if fit_peak > 0:
                    IP, CR = fit_peak, fit_peak / (spectrum.live_time / 1E12)
                else:
                    IP, CR = int_peak, int_peak / (spectrum.live_time / 1E12)

                if LFC:
                    ast = '*'
                else:
                    ast = ''

                _title.set_text(rf"{spectrum_counter:03d} | real: {spectrum.real_time / 1E12:.0f} s (DT{ast}: {1-spectrum.live_time/spectrum.real_time:.1f} %), NET: {IP:.0f} (CR: {CR:.1f} s$^{{-1}}$)")

                fitY = _peakfit(X, *popt)
                BKG = _BKGfit(X, *popt)

                _line_spectrum[0].set_ydata(spectrum.counts[X] + 0.001)
                _line_fit[0].set_ydata(fitY)
                _line_background[0].set_ydata(BKG)

                res_fitY = (fitY-(spectrum.counts[X] + 0.001))/np.sqrt(spectrum.counts[X] + 0.001)

                _res_fit[0].set_ydata(res_fitY)

                ax.set_ylim(0, np.max(spectrum.counts[X])*1.05+10)
                axres.set_ylim(safelim(res_fitY, 3.1, sign='-'), safelim(res_fitY, 3.1, sign='+'))
                fig.tight_layout()
                plt.draw()
                plt.pause(0.25)

            #check condition 1, 2 or 3
            if cond_PK_UNC is not None and urfit_peak < cond_PK_UNC:
                spectrum.extended_dead_time = str(extended_deadtime)
                if LFC:
                    spectrum.counts = np.round(spectrum.counts).astype(int)

                print(f'spectrum {spectrum_counter:03d} terminated')
                spectrum, spectrum_counter, reference_date, sinfo, spectra_collection = _make_spectrum(spectrum, spectrum_counter, reference_date, fit_peak, fit_peak*urfit_peak/100, int_peak, int_peak*urint_peak/100, pup_corr, upup_corr, centr=centr, source=source, spectrum_collection=spectra_collection)
                decay_data.append(sinfo)

            elif cond_RT is not None and spectrum.real_time / 1E12 >= cond_RT:

                if hard_time_limit:
                    return _produce_decay_dataframe(decay_data, source=source)

                spectrum.extended_dead_time = str(extended_deadtime)
                if LFC:
                    spectrum.counts = np.round(spectrum.counts).astype(int)

                print(f'spectrum {spectrum_counter:03d} terminated')
                spectrum, spectrum_counter, reference_date, sinfo, spectra_collection = _make_spectrum(spectrum, spectrum_counter, reference_date, fit_peak, fit_peak*urfit_peak/100, int_peak, int_peak*urint_peak/100, pup_corr, upup_corr, centr=centr, source=source, spectrum_collection=spectra_collection)
                decay_data.append(sinfo)

            elif cond_LT is not None and spectrum.live_time / 1E12 >= cond_LT:
                spectrum.extended_dead_time = str(extended_deadtime)
                if LFC:
                    spectrum.counts = np.round(spectrum.counts).astype(int)

                print(f'spectrum {spectrum_counter:03d} terminated')
                spectrum, spectrum_counter, reference_date, sinfo, spectra_collection = _make_spectrum(spectrum, spectrum_counter, reference_date, fit_peak, fit_peak*urfit_peak/100, int_peak, int_peak*urint_peak/100, pup_corr, upup_corr, centr=centr, source=source, spectrum_collection=spectra_collection)
                decay_data.append(sinfo)

    spectrum.extended_dead_time = str(extended_deadtime)
    if LFC:
        spectrum.counts = np.round(spectrum.counts).astype(int)
    spectrum_counter = -1

    if not exclude_last:
        spectrum, spectrum_counter, reference_date, sinfo, spectra_collection = _make_spectrum(spectrum, spectrum_counter, reference_date, fit_peak, fit_peak*urfit_peak/100, int_peak, int_peak*urint_peak/100, pup_corr, upup_corr, centr=centr, source=source, spectrum_collection=spectra_collection)
        decay_data.append(sinfo)
        print(f'{spectrum_counter:03d}')
    else:
        print(f'{spectrum_counter:03d} (excluded)')

    return _produce_decay_dataframe(decay_data, source=source), spectra_collection


def manage_stream_digitized_data_time(filename, chunksize=100000, channel_signal=None, extended_deadtime=100, pileuprejection_limit=0.5, spectrum_channels=16384, verbose=True, LFC=False, LFC_type='ICR', condition_options={}, peak_animation=True, source=0, exclude_last=False, two_tau=0.0, two_tau_unc=0.0, fit=''):#maybe loss-free counting (LFC)
    """Save a series of spectra from the stream of a single list mode file and return a summary dataframe; much faster than the one dedicated to peak uncertainty and more consistent on acquisition times

    Parameters
    ----------
    filename : str (path)
        name of the gamma spectrum in list mode file
    chunksize : int
        number of rows to elaborate at each iteration; corresponds to the resolution of the peak (default 100000)
    channel_signal : int
        index of the channel to be analyzed; if the list mode file comes from a single channel this could be neglected (default None)
    extended_deadtime : float
        time in us (default 100)
    pileuprejection_limit : float
        time limit in which two successive triggers are both excluded, in microseconds (default 0.5)
    spectrum_channels : int
        number of channels in the spectrum (default 16384)
    verbose : bool
        display information during elaboration (default True)
    LFC : bool
        whether perform elaboration in loss-free counts method, useful for spectra with rapidly changing dead times (default False)
    LFC_type : str
        kind of correction to perform to obtain 0 dead time (default 'ICR').
        valid options are:
        'ICR'  : correction based on the input count rate (1 - t * ICR)
        'time' : correction based on ratio between real and live times (real / live)
    condition_options : dict
        descr
    peak_animation : bool
        whether show the peak fitting in the selected ROI (default True)
    source : int
        id for the stream data, a nice way to discriminate between multiple decays for the same isotope (default 0)
    exclude_last : bool
        whether to exclude the last chunk of incomplete information (default False)
    two_tau : float
        2t experimental value in microseconds defined as the gradient of lnA versus R (default 0.0)
    two_tau_unc : float
        standard uncertainty of two_tau in microseconds (default 0.0)
    fit : str
        code to recall a specific fit from peaks module (default "")

    Return
    ------
    decay_data_df : pd.DataFrame
        dataframe containing relevant information of all the saved spectra
    spectra_collection : list
        list of spectrum objects obtained from the elaboration
    """

    cond_RT = condition_options.get('real_time', None)
    cond_ROI = condition_options.get('roi', (0, spectrum_channels-1))
    cond_PK = condition_options.get('centroid', -1)
    cond_BKG_CH = condition_options.get('bkg_nch', (10, 10))
    decay_data = []
    spectra_collection = []

    if cond_RT is None:
        raise ValueError('condition for "real_time" is required')

    #fit selection
    _peakfit, _BKGfit = pk_elab._get_fit(fit)

    #date information
    date_from_file = retrieve_date(filename)

    #for pileup correction factor
    try:
        two_tau, two_tau_unc = float(two_tau)/1E6, float(two_tau_unc)/1E6
    except (TypeError, ValueError):
        two_tau, two_tau_unc = 0.0, 0.0

    with pd.read_csv(filename, sep=';', header=0, names=('BOARD', 'CHANNEL', 'TIMETAG', 'ENERGY', 'FLAGS', 'PROBE_CODE'), chunksize=chunksize, comment='#') as spectrum_file:
        #initialize the spectrum
        spectrum = GammaSpectrum(channels=spectrum_channels, var=np.zeros(spectrum_channels))
        spectrum.extended_dead_time = str(extended_deadtime)

        _previous_endpoint = 0
        reference_date = date_from_file if date_from_file is not None else datetime.datetime.today()
        spectrum_counter = 0

        #default fit parameters
        X = np.arange(cond_ROI[0], cond_ROI[1]+1)

        if peak_animation:
            plt.ion()
            fig, ax = plt.subplots(nrows=1, ncols=1)
            yinit = np.array([np.nan for _ in X])
            _line_spectrum = ax.plot(X, yinit, linestyle='-', color='grey', linewidth=1, marker='')
            _title = ax.set_title(f"{spectrum_counter:03d} | real: {spectrum.real_time / 1E12:.0f} s")
            ax.set_xlim(np.min(X), np.max(X))

        #initial condition
        realtimelimit = _previous_endpoint + cond_RT * 1E12

        for chunk in spectrum_file:

            #no need to filter the signal by channel again, but just to be sure
            if isinstance(channel_signal, int):
                _filter_signal = chunk['CHANNEL'] == channel_signal
            else:
                _filter_signal = np.ones(len(chunk['CHANNEL']), dtype=bool)
            chunk = chunk[_filter_signal]

            while realtimelimit < np.max(chunk['TIMETAG']):
                _filter_time = chunk['TIMETAG'] < realtimelimit

                local_ch = SpectrumChunkPHA([(trigger_time, energy, flag) for trigger_time, energy, flag in zip(chunk[_filter_time]['TIMETAG'], chunk[_filter_time]['ENERGY'], chunk[_filter_time]['PROBE_CODE'])], extended_deadtime, pileuprejection_limit, _previous_endpoint, channels=spectrum.channels, LFC=LFC, LFC_type=LFC_type)

                _previous_endpoint = local_ch.end
                #add chunk's information to the total spectrum
                spectrum += local_ch
                if verbose:
                    if LFC:
                        comment = 'LFC'
                    else:
                        comment = ''
                    print(f'REAL: {spectrum.real_time / 1E12:.1f} s ({(1-spectrum.live_time/spectrum.real_time)*100:.1f} %) {comment}')

                spectrum.extended_dead_time = str(extended_deadtime)
                if LFC:
                    spectrum.counts = np.round(spectrum.counts).astype(int)

                if cond_RT * 1E12 > spectrum.real_time:
                    liveadd = cond_RT * 1E12 - spectrum.real_time
                    spectrum.real_time += liveadd
                    spectrum.live_time += liveadd

                fit_peak, urfit_peak = 0.0, 1.0
                int_peak, urint_peak = 0.0, 1.0
                pup_corr, upup_corr = 1.0, 0.0
                centr = 0

                #check all conditions anyway?
                if cond_ROI[1] > cond_PK and cond_ROI[0] < cond_PK:
                    Y = spectrum.counts[X] + 0.001
                    try:
                        Y2 = spectrum.var[X] + 0.001
                    except Exception:
                        Y2 = None
                    popt, pcov = pk_elab._manage_fit(X, Y, cond_PK, _peakfit, Y2=spectrum.var)

                    if popt[0] is not None:
                        fit_peak = popt[0]
                        urfit_peak = (np.sqrt(np.diag(pcov))[0] / fit_peak) * 100
                    else:
                        fit_peak = 0
                        urfit_peak = 100

                    spcICR, uspcICR = spectrum.input_count_rate()
                    pup_corr, upup_corr = pk_elab.evaluate_pileup_correction(spcICR, uspcICR, two_tau, two_tau_unc)
                    centr = popt[1]

                if cond_BKG_CH[0] + cond_BKG_CH[1] < cond_ROI[1] - cond_ROI[0]:
                    int_peak, urint_peak = pk_elab._peak_integral(spectrum, cond_ROI, cond_BKG_CH)

                if peak_animation:

                    CR = spectrum.defined_spectrum_integral(X[0], X[-1]) / (spectrum.live_time / 1E12)

                    if LFC:
                        ast = '*'
                    else:
                        ast = ''

                    _title.set_text(rf"{spectrum_counter:03d} | real: {spectrum.real_time / 1E12:.0f} s (DT{ast}: {1-spectrum.live_time/spectrum.real_time:.1f} %), CR: {CR:.1f} s$^{{-1}}$")

                    _line_spectrum[0].set_ydata(spectrum.counts[X] + 0.001)

                    ax.set_ylim(0, np.max(spectrum.counts[X])*1.05+10)
                    fig.tight_layout()
                    plt.draw()
                    plt.pause(0.25)

                print(f'spectrum {spectrum_counter:03d} terminated')
                spectrum, spectrum_counter, reference_date, sinfo, spectra_collection = _make_spectrum(spectrum, spectrum_counter, reference_date, fit_peak, fit_peak*urfit_peak/100, int_peak, int_peak*urint_peak/100, pup_corr, upup_corr, centr=centr, source=source, spectrum_collection=spectra_collection)
                decay_data.append(sinfo)

                chunk = chunk[~_filter_time]
                realtimelimit = _previous_endpoint + cond_RT * 1E12


            _filter_time = np.ones(len(chunk['TIMETAG']), dtype=bool)

            local_ch = SpectrumChunkPHA([(trigger_time, energy, flag) for trigger_time, energy, flag in zip(chunk[_filter_time]['TIMETAG'], chunk[_filter_time]['ENERGY'], chunk[_filter_time]['PROBE_CODE'])], extended_deadtime, pileuprejection_limit, _previous_endpoint, channels=spectrum.channels, LFC=LFC, LFC_type=LFC_type)

            _previous_endpoint = local_ch.end
            #add chunk's information to the total spectrum
            spectrum += local_ch
            if verbose:
                if LFC:
                    comment = 'LFC'
                else:
                    comment = ''
                print(f'REAL: {spectrum.real_time / 1E12:.1f} s ({(1-spectrum.live_time/spectrum.real_time)*100:.1f} %) {comment}')

            if peak_animation:

                CR = spectrum.defined_spectrum_integral(X[0], X[-1]) / (spectrum.live_time / 1E12)

                if LFC:
                    ast = '*'
                else:
                    ast = ''

                _title.set_text(rf"{spectrum_counter:03d} | real: {spectrum.real_time / 1E12:.0f} s (DT{ast}: {1-spectrum.live_time/spectrum.real_time:.1f} %), CR: {CR:.1f} s$^{{-1}}$")

                _line_spectrum[0].set_ydata(spectrum.counts[X] + 0.001)

                ax.set_ylim(0, np.max(spectrum.counts[X])*1.05+10)
                fig.tight_layout()
                plt.draw()
                plt.pause(0.25)

    spectrum.extended_dead_time = str(extended_deadtime)
    if LFC:
        spectrum.counts = np.round(spectrum.counts).astype(int)

    fit_peak, urfit_peak = 0.0, 1.0
    int_peak, urint_peak = 0.0, 1.0
    pup_corr, upup_corr = 1.0, 0.0
    centr = 0

    #check all conditions anyway?
    if cond_ROI[1] > cond_PK and cond_ROI[0] < cond_PK:
        Y = spectrum.counts[X] + 0.001
        try:
            Y2 = spectrum.var[X] + 0.001
        except Exception:
            Y2 = None
        popt, pcov = pk_elab._manage_fit(X, Y, cond_PK, _peakfit, Y2=spectrum.var)
        if popt[0] is not None:
            fit_peak = popt[0]
            urfit_peak = (np.sqrt(np.diag(pcov))[0] / fit_peak) * 100
        else:
            fit_peak = 0
            urfit_peak = 100

        spcICR, uspcICR = spectrum.input_count_rate()
        pup_corr, upup_corr = pk_elab.evaluate_pileup_correction(spcICR, uspcICR, two_tau, two_tau_unc)
        centr = popt[1]

    if cond_BKG_CH[0] + cond_BKG_CH[1] < cond_ROI[1] - cond_ROI[0]:
        int_peak, urint_peak = pk_elab._peak_integral(spectrum, cond_ROI, cond_BKG_CH)

    spectrum_counter = -1

    if not exclude_last:
        spectrum, spectrum_counter, reference_date, sinfo, spectra_collection = _make_spectrum(spectrum, spectrum_counter, reference_date, fit_peak, fit_peak*urfit_peak/100, int_peak, int_peak*urint_peak/100, pup_corr, upup_corr, centr=centr, source=source, spectrum_collection=spectra_collection)
        decay_data.append(sinfo)
        print(f'{spectrum_counter:03d}')
    else:
        print(f'{spectrum_counter:03d} (excluded)')

    return _produce_decay_dataframe(decay_data, source=source), spectra_collection

def open_digitized_spectrumPHA(filename, chunksize=100000, channel_signal=None, extended_deadtime=100, pileuprejection_limit=0.5, spectrum_channels=16384, verbose=True, LFC=False, LFC_type='ICR'):#maybe loss-free counting (LFC)
    """Return a time value (converted in s) from the corresponding dataframe line (single channel is assumed)

    Parameters
    ----------
    filename : str (path)
        name of the gamma spectrum in list mode file
    chunksize : int
        number of rows to elaborate at each iteration; useful for huge files (default 100000)
    channel_signal : int
        index of the channel to be analyzed; if the list mode file comes from a single channel this could be neglected (default None)
    extended_deadtime : float
        time in us (default 100)
    pileuprejection_limit : float
        time limit in which two successive triggers are both excluded, in microseconds (default 0.5)
    spectrum_channels : int
        number of channels in the spectrum (default 16384)
    verbose : bool
        display information during elaboration (default True)
    LFC : bool
        whether perform elaboration in loss-free counts method, useful for spectra with rapidly changing dead times (default False)
    LFC_type : str
        kind of correction to perform to obtain 0 dead time (default 'ICR').
        valid options are:
        'ICR'  : correction based on the input count rate (1 - t * ICR)
        'time' : correction based on ratio between real and live times (real / live)

    Return
    ------
    spectrum : GammaSpectrum
        elaborated spectrum
    """
    #date information
    date_from_file = retrieve_date(filename)
    Ecalib_fromfile = retrieve_Ecalibration(filename)

    with pd.read_csv(filename, sep=';', header=0, names=('BOARD', 'CHANNEL', 'TIMETAG', 'ENERGY', 'FLAGS', 'PROBE_CODE'), chunksize=chunksize, comment='#') as spectrum_file:
        #initialize the spectrum
        spectrum = GammaSpectrum(channels=spectrum_channels, var=np.zeros(spectrum_channels))
        spectrum.extended_dead_time = str(extended_deadtime)

        _previous_endpoint = 0

        for chunk in spectrum_file:

            #no need to filter the signal by channel again, but just to be sure
            if isinstance(channel_signal, int):
                _filter_signal = chunk['CHANNEL'] == channel_signal
            else:
                _filter_signal = np.ones(len(chunk['CHANNEL']), dtype=bool)

            local_ch = SpectrumChunkPHA([(trigger_time, energy, flag) for trigger_time, energy, flag in zip(chunk[_filter_signal]['TIMETAG'], chunk[_filter_signal]['ENERGY'], chunk[_filter_signal]['PROBE_CODE'])], extended_deadtime, pileuprejection_limit, _previous_endpoint, channels=spectrum.channels, LFC=LFC, LFC_type=LFC_type)

            _previous_endpoint = local_ch.end
            #add chunk's information to the total spectrum
            spectrum += local_ch
            if verbose:
                if LFC:
                    comment = 'LFC'
                else:
                    comment = ''
                print(f'REAL: {spectrum.real_time / 1E12:.1f} s ({(1-spectrum.live_time/spectrum.real_time)*100:.1f} %) {comment}')

    spectrum.extended_dead_time = str(extended_deadtime)
    if LFC:
        spectrum.counts = np.round(spectrum.counts).astype(int)

    spectrum.datetime = date_from_file if date_from_file is not None else datetime.datetime(2000, 1, 1, 12, 0, 0)
    spectrum.E_cal = Ecalib_fromfile

    spectrum.o_filename = filename

    return spectrum

def open_digitized_spectrumFPGA(filename, chunksize=100000, channel_signal=None, extended_deadtime=100, pileuprejection_limit=0.5, ADC_conv=32, unc_trigger=1, spectrum_channels=16384, verbose=True, LFC=False, LFC_type='ICR'):#maybe loss-free counting (LFC)
    """Return a time value (converted in s) from the corresponding dataframe line (single channel is assumed)

    Parameters
    ----------
    filename : str (path)
        name of the gamma spectrum in list mode file
    chunksize : int
        number of rows to elaborate at each iteration; useful for huge files (default 100000)
    channel_signal : int
        index of the channel to be analyzed; if the list mode file comes from a single channel this could be neglected (default None)
    extended_deadtime : float
        time in us (default 100)
    pileuprejection_limit : float
        time limit in which two successive triggers are both excluded, in microseconds (default 0.5)
    spectrum_channels : int
        number of channels in the spectrum (default 16384)
    verbose : bool
        display information during elaboration (default True)
    LFC : bool
        whether perform elaboration in loss-free counts method, useful for spectra with rapidly changing dead times (default False)
    LFC_type : str
        kind of correction to perform to obtain 0 dead time (default 'ICR').
        valid options are:
        'ICR'  : correction based on the input count rate (1 - t * ICR)
        'time' : correction based on ratio between real and live times (real / live)

    Return
    ------
    spectrum : GammaSpectrum
        elaborated spectrum
    """
    #date information
    date_from_file = None#retrieve_date(filename)
    #Ecalib_fromfile = retrieve_Ecalibration(filename)

    with pd.read_csv(filename, sep=',', header=0, names=('CHARGE', 'BUSY', 'IDLE', 'TAG'), chunksize=chunksize, comment='#') as spectrum_file:
        #initialize the spectrum
        spectrum = GammaSpectrum(channels=spectrum_channels, var=np.zeros(spectrum_channels))
        spectrum.extended_dead_time = str(extended_deadtime)

        _previous_endpoint = 0

        for chunk in spectrum_file:

            local_ch = SpectrumChunkFPGA(chunk, _previous_endpoint, extended_deadtime, ADC_conv=ADC_conv, unc_trigger=unc_trigger, channels=spectrum.channels, LFC=LFC, LFC_type=LFC_type)

            _previous_endpoint = local_ch.end
            #add chunk's information to the total spectrum
            spectrum += local_ch
            if verbose:
                if LFC:
                    comment = 'LFC'
                else:
                    comment = ''
                print(f'REAL: {spectrum.real_time / 1E12:.1f} s ({(1-spectrum.live_time/spectrum.real_time)*100:.1f} %) {comment}')

    spectrum.extended_dead_time = str(extended_deadtime)
    if LFC:
        spectrum.counts = np.round(spectrum.counts).astype(int)

    spectrum.datetime = date_from_file if date_from_file is not None else datetime.datetime(2000, 1, 1, 12, 0, 0)
    #spectrum.E_cal = Ecalib_fromfile

    spectrum.o_filename = filename

    return spectrum

def spectrum_multielab(filename, extended_deadtimes, n_channels, chunksize=250000, pileuprejection_limit=0.5, folder='elaboration', verbose=True, LFC=False, LFC_type='ICR', full_output=True, save_format='n42'):
    """Elaboration of the same list mode spectrum with multiple extended dead times
    
    Parameters
    ----------
    filename : str (path)
        name of the gamma spectrum in list mode file
    extended_deadtimes : list
        list of floats indicating extended dead times in μs to feed the elaboration
    n_channels : int
        number of channels in the spectrum
    chunksize : int
        number of rows to elaborate at each iteration; useful for huge files (default 250000)
    pileuprejection_limit : float
        time limit in which two successive triggers are both excluded, in microseconds (default 0.5)
    folder : str (or Path)
        name of the folder where the output is saved (default 'elaboration')
    verbose : bool
        display information during elaboration (default True)
    LFC : bool
        whether perform elaboration in loss-free counts method, useful for spectra with rapidly changing dead times (default False)
    LFC_type : str
        kind of correction to perform to obtain 0 dead time (default 'ICR').
        valid options are:
        'ICR'  : correction based on the input count rate (1 - t * ICR)
        'time' : correction based on ratio between real and live times (real / live)
    full_output : bool
        whether save also a csv summary of the chunk's information (default True)
    save_format : str
        format to save the resulting spectra, options ["n42", "asc"] (default "n42")
    
    Return
    ------
    None
    """

    for EXTENDED_DEADTIME in extended_deadtimes:
        spectrum = open_digitized_spectrumPHA(filename, chunksize=chunksize, channel_signal=None, extended_deadtime=EXTENDED_DEADTIME, pileuprejection_limit=pileuprejection_limit, spectrum_channels=n_channels, verbose=verbose, LFC=LFC, LFC_type=LFC_type)
        
        if not os.path.isdir(folder):
            os.makedirs(folder)
        path, file_name = os.path.split(filename)
        fname, extension = os.path.splitext(file_name)

        if save_format.lower() == 'asc':
            spectrum._save(os.path.join(folder, f'{fname}_EDT{EXTENDED_DEADTIME}'), full=full_output)
        else:
            spectrum._saven42(os.path.join(folder, f'{fname}_EDT{EXTENDED_DEADTIME}'), full=full_output)
        
        print(f'EXTENDED DEAD TIME {EXTENDED_DEADTIME} μs COMPLETED!')
        
def multi_spectrum_elab(filenames, extended_deadtimes, n_channels, chunksize=250000, pileuprejection_limit=0.5, folder='elaboration', verbose=True, LFC=False, LFC_type='ICR', full_output=True, save_format='n42'):
    """Elaboration of multiple list mode spectra with multiple extended dead times
    
    Parameters
    ----------
    filenames : list
        list of names for the gamma spectra in list mode file
    extended_deadtimes : list
        list of floats indicating extended dead times in μs to feed the elaboration
    n_channels : int
        number of channels in the spectra, assumed to be the same for everyone
    chunksize : int
        number of rows to elaborate at each iteration; useful for huge files and LFC (default 250000)
    pileuprejection_limit : float
        time limit in which two successive triggers are both excluded, in microseconds (default 0.5)
    folder : str (or Path)
        name of the folder where the output is saved (default 'elaboration')
    verbose : bool
        display information during elaboration (default True)
    LFC : bool
        whether perform elaboration in Loss-Free Counts method (LFC), useful for spectra with rapidly changing dead times (default False)
    LFC_type : str
        kind of correction to perform to obtain 0 dead time (default 'ICR').
        valid options are:
        'ICR'  : correction based on the input count rate (1 - t * ICR)
        'time' : correction based on ratio between real and live times (real / live)
    full_output : bool
        whether save also a csv summary of the chunk's information (default True)
    save_format : str
        format to save the resulting spectra, options ["n42", "asc"] (default "n42")
    
    Return
    ------
    None
    """

    for _file in filenames:
        try:
            print(f'FILE ... {_file}')
            spectrum_multielab(_file, extended_deadtimes, n_channels, chunksize=chunksize, pileuprejection_limit=pileuprejection_limit, folder=folder, verbose=verbose, LFC=LFC, LFC_type=LFC_type, full_output=full_output)
            print(f'FILE {_file} COMPLETED!')
        except (IOError, pd.errors.ParserError):
            print('FILE ERROR')
            
def get_spectra_filenames(data_folder):
    """Get spectra names from folder
    
    Parameters
    ----------
    data_folder : str (Path)
        folder where gamma spectra are found
    
    Return
    ------
    filenames : list
        filenames of .csv files in data_folder
    """
    return [os.path.join(data_folder, filename) for filename in os.listdir(data_folder) if filename.lower().endswith('.csv') and os.path.isfile(os.path.join(data_folder, filename))]

def open_spectrum_file(filename):
    """Get spectra names from folder

    Parameters
    ----------
    filename : str (Path)
        filepath where gamma spectra information are found (.asc or .n42 formats)

    Return
    ------
    gammaspectrum : GammaSpectrum
        spectrum object build from the information gathered in filename
    """
    if filename.lower().endswith('.asc'):
        real, live, spectrum_counts, real_unc, live_unc, startcounting, ext_deadtime = openASCfile(filename)
    elif filename.lower().endswith('.n42'):
        real, live, spectrum_counts, real_unc, live_unc, startcounting, ext_deadtime = open_n42_file(filename)
    else:
        raise NotImplementedError('only ".asc" or ".n42" are readable from this function')

    return GammaSpectrum(real_time=real, live_time=live, counts=spectrum_counts, real_time_unc=real_unc, live_time_unc=live_unc, channels=len(spectrum_counts), chunk_info=None, extended_dead_time=ext_deadtime, var=None)

if __name__ == '__main__':
    fname = 'Data_CH0@DT5730SB_27099_Pb-212-0.csv'
    spc = open_digitized_spectrumPHA(fname)
    Rs = [pk_elab.Region(i[0], i[1], i[2]) for i in ([1250,1320,1287],)]
    pk_elab.peak_elaboration(spc, Rs)
