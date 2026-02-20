"""Script to perform on gamma list mode spectrum

description

it contains functions:
open_digitized_spectrum, openASCfile

it contains classes:
GammaSpectrum, SpectrumChunk, ElaborationSpectrum

This module can be imported into another script with:
"from gammaspec import spectra"
giving access to all its methods and classes

author: Marco Di Luzio
email: ...
"""

#imports
import os
import pickle
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import peaks as pk_elab

from matplotlib.gridspec import GridSpec


class Calibration:
    #provisional calibration just used for testing
    def __init__(self):
        self.name = 'DUMMY CALIBRATION FOR TESTING'
        self.energy_params = np.array([0.02618033, 0.05337198])
        self.fwhm_params = np.array([0.015, 220])


class GammaSpectrum:#consider also loss-free counting (LFC)
    """Class for reconstructed gamma spectrum
    
    Attributes
    ----------
    self.real_time : int
        real time of the spectrum in fs
    self.live_time : int
        live time of the spectrum in fs
    self.real_time_unc : int
        real time uncertainty of the spectrum in fs
    self.live_time_unc : int
        live time uncertainty of the spectrum in fs
    self.channels : int
        number of channels of the spectrum (property, cannot be modified after initialized)
    self.counts : np.array
        array of int identifying counts of the spectrum for each channel
    self.extended_dead_time : str
        extended dead time in microseconds; if '-1' info not available
    self.chunk_info : list
        descr
    
    Methods
    -------
    __add__()
        perform sum of spectra, overrides the defult function
    _save()
        save class information into a text file
    _get_chunks_info()
        return information of spectrum chunks as a pd.DataFrame
    defined_spectrum_integral()
        calculate the integral of a defined spectrum slice
    """
    def __init__(self, real_time=0, live_time=0, counts=None, real_time_unc=0, live_time_unc=0, channels=16384, chunk_info=None, extended_dead_time='-1'):
        """GammaSpectrum class constructor
        
        Parameters
        ----------
        real_time : int
            real time of the spectrum in fs (default 0)
        live_time : int
            live time of the spectrum in fs (default 0)
        counts : np.array
            array of int identifying the number of counts for any channel, if None an array of zeros is created (default None)
        real_time_unc : int
            real time standard uncertainty in fs (default 0)
        live_time_unc : int
            live time standard uncertainty in fs (default 0)
        channels : int
            number of channels of spectrum (default 16384)
        chunk_info : list
            descr
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
        return GammaSpectrum(real_time, live_time, counts, real_time_unc, live_time_unc, len(counts), chunk_info_sum)
        
    def _get_chunks_info(self):
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
        with open(asc_filename, 'w') as wfile:
            for count in self.counts:
                wfile.write(f'{count}\n')
            wfile.write(f'#LiveTime={self.live_time / 1E12:.6f}\n')
            wfile.write(f'#LiveTime_unc={self.live_time_unc / 1E12:.6f}\n')
            wfile.write(f'#TrueTime={self.real_time / 1E12:.6f}\n')
            wfile.write(f'#TrueTime_unc={self.real_time_unc / 1E12:.6f}\n')
            wfile.write(f'#AcqStart={"2000-01-01T00:00:00"}\n')
            wfile.write(f'#ExtDeadTime={self.extended_dead_time}\n')
        if full:
            df = self._get_chunks_info()
            df.to_csv(csv_filename)


class SpectrumChunk(GammaSpectrum):#non-overlapping spectrum chunks, #consider also loss-free counting (LFC)
    """Class to manage non-overlapping chunks of information of the gamma spectrum (inherits from GammaSpectrum)
    
    Attributes
    ----------
    self.deadtimes : list
        descr
    self.gatetimes : list
        descr
    self.start : int
        starting timestamp of the chunk
    self.end : int
        finishing timestamp of the chunk
    self.gate_close : bool
        descr
    
    Methods
    -------
    _count_counting_time()
        descr
    _count_counts()
        descr
    _extend_deadtime()
        descr
    _manage_gatetime()
        descr
    _add_deadtime_to_trigger()
        descr
    _add_deadtime_to_trigger_flagsaware()
        descr
    _add_deadtime_to_trigger_regardless_of_flag()
        descr
    """
    def __init__(self, trigger_times, extended_deadtime, gate_times, _previous_endpoint, _gate_close, unc_trigger=1, channels=16384, LFC=False):
        """SpectrumChunk class constructor
        
        Parameters
        ----------
        trigger_times : list
            list of 3-items tuple for triggers (time, energy, flag)
        extended_deadtime : float
            extended dead time to apply to triggers, in microseconds
        gate_times : list
            list of 2-items tuple for gate triggers (time, flag)
        _previous_endpoint : int
            timestamp indicating the endpoint of the previous chunk
        _gate_close : bool
            whether the gate from the end of previous chunk was close (dead time) or not (live time)
        unc_trigger : float
            uncertainty of each timespan in microseconds (default 1)
        channels : int
            number of channels of spectrum (default 16384)
        LFC : bool
            whether perform elaboration in loss-free counts method, useful for spectra with rapidly changing dead times (default False)
            
        Return
        ------
        None        
        """
        super().__init__(channels=channels)
        #conversion of times from micro to femtoseconds
        extended_deadtime = int(extended_deadtime * 1E6)
        unc_trigger = int(unc_trigger * 1E6)
        
        chunk_info = {}
        chunk_info[('original_trigger', 'length')] = len(trigger_times)
        chunk_info[('original_gate', 'length')] = len(gate_times)

        _trigger_codes = pd.Series([row[2] for row in trigger_times], dtype=object)
        _trigger_codes = _trigger_codes.unique()
        
        _gate_codes = pd.Series([row[1] for row in gate_times], dtype=object)
        _gate_codes = _gate_codes.unique()
        
        chunk_info[('original_trigger', 'flags')] = _trigger_codes
        chunk_info[('original_gate', 'flags')] = _gate_codes
        
        self.start = _previous_endpoint
        self.deadtimes = self._add_deadtime_to_trigger_selection(trigger_times, extended_deadtime)
        self.gatetimes = self._add_deadtime_to_trigger_flagsaware(gate_times, _gate_close)
        
        #extend dead time for triggers
        self._extend_deadtime()
        
        dttr = np.array([_row[1] - _row[0] for _row in self.deadtimes])
        if len(dttr) == 0:
            dttr_min, dttr_max, dttr_ave = np.nan, np.nan, np.nan
        else:
            dttr_min, dttr_max, dttr_ave = np.min(dttr), np.max(dttr), np.average(dttr)
            
        dtgt = np.array([_row[1] - _row[0] for _row in self.gatetimes])
        if len(dtgt) == 0:
            dtgt_min, dtgt_max, dtgt_ave = np.nan, np.nan, np.nan
        else:
            dtgt_min, dtgt_max, dtgt_ave = np.min(dtgt), np.max(dtgt), np.average(dtgt)
        chunk_info[('elaborated_trigger', 'length')] = len(self.deadtimes)
        chunk_info[('elaborated_trigger', 'min')] = dttr_min / 1E6
        chunk_info[('elaborated_trigger', 'average')] = dttr_ave / 1E6
        chunk_info[('elaborated_trigger', 'max')] = dttr_max / 1E6
        chunk_info[('elaborated_gate', 'length')] = len(self.gatetimes)
        chunk_info[('elaborated_gate', 'min')] = dtgt_min / 1E6
        chunk_info[('elaborated_gate', 'average')] = dtgt_ave / 1E6
        chunk_info[('elaborated_gate', 'max')] = dtgt_max / 1E6
        
        if len(self.deadtimes) > 0 and len(self.gatetimes) > 0:
            self.end = int(np.max((self.deadtimes[-1][1], self.gatetimes[-1][1])))
            if self.end == self.gatetimes[-1][1]:
                self.gate_close = True
            else:
                self.gate_close = False
        elif len(self.deadtimes) == 0 and len(self.gatetimes) == 0:
            self.end = self.start
            self.gate_close = False
        elif len(self.deadtimes) > 0:
            self.end = int(self.deadtimes[-1][1])
            self.gate_close = False
        else:
            self.end = int(self.gatetimes[-1][1])
            self.gate_close = True
        
        #intersection
        self.deadtimes, self.gatetimes = self._intersection(_previous_endpoint, _gate_close)
        
        dtgt = np.array([_row[1] - _row[0] for _row in self.gatetimes])
        if len(dtgt) == 0:
            dtgt_min, dtgt_max, dtgt_ave = np.nan, np.nan, np.nan
        else:
            dtgt_min, dtgt_max, dtgt_ave = np.min(dtgt), np.max(dtgt), np.average(dtgt)
        chunk_info[('elaborated_deadtime', 'length')] = len(self.gatetimes)
        chunk_info[('elaborated_deadtime', 'min')] = dtgt_min
        chunk_info[('elaborated_deadtime', 'average')] = dtgt_ave
        chunk_info[('elaborated_deadtime', 'max')] = dtgt_max

        self.real_time, self.live_time, self.real_time_unc, self.live_time_unc = self._count_counting_time(unc=unc_trigger)
        self.counts = self._count_counts()
        
        #HERE THE THING
        if LFC:
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
        chunk_info['gate'] = self.gate_close
        chunk_info['LFC'] = LFC
        
        self.chunk_info.append(pd.Series(chunk_info))
        
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
            list of 4-items tuple for triggers (time, time + extended dead time, energy, flag)
        """
        return [(row[0], row[0] + dead_time, row[1], row[2]) for row in signal_sequence]#if row[2] not in ()
        
    def _add_deadtime_to_trigger_flagsaware(self, signal_sequence, _gate_close):
        """create couples (start_time, end_time) for triggers of gates where start_time and end_time are identified by their corresponding flags
        
        Parameters
        ----------
        signal_sequence : list
            list of 3-items tuple for triggers (time, energy, flag)
        _gate_close : bool
            whether the previous spectrum chunk ended with gate closed (dead time) or open
            
        Return
        ------
        gatetimes : list
            list of 2-items tuple for triggers (time start dead time, time end dead time)
        """
        start_id = ('0x400', '0xc00')#WATCH HERE!!!
        end_id = ('0x0', '0x800')
        
        gatetimes = []
        
        if _gate_close:
            previous = self.start
        else:
            previous = None

        #check the logic here!
        
        idx = 0
        while True:
            try:
                line = signal_sequence[idx]
            except IndexError:
                if previous is not None:
                    closure = np.max([self.deadtimes[-1][1], previous + 1])
                    gatetimes.append((previous, closure))
                break
            else:
                if line[1] in start_id and previous is None:
                    previous = line[0]
                    
                elif line[1] in end_id and previous is not None:
                    gatetimes.append((previous, line[0]))
                    previous = None

                elif line[1] in start_id and previous is not None:
                    #print('double start situation')#what to do here? disregard? open a different thread?
                    previous = line[0]
                    
                elif line[1] in end_id and previous is None:
                    pass#print('end without start situation')#what to do here? disregard? open a different thread?
                    
                else:
                    print('ERROR!')
                    
                idx += 1

        return gatetimes
        
    def _add_deadtime_to_trigger_regardless_of_flag(signal_sequence, dead_time):
        """create couples (start-time, end-time) for triggers of gates where end-time is start-time + some fixed time
        
        Parameters
        ----------
        signal_sequence : list
            list of 3-items tuple for triggers (time, energy, flag)
        dead_time : int
            extended dead time to apply to gate times, in femtoseconds
            
        Return
        ------
        gatetimes : list
            list of 2-items tuple for triggers (time start dead time, time end dead time)
        """
        return [(row[0], row[0] + dead_time) for row in signal_sequence]
        
    def _count_counting_time(self, unc=1):
        """create couples (start_time, end_time) for triggers of gates where end_time = start_time + some_fixed_time
        
        Parameters
        ----------
        unc : int
            standard uncertainty on the single timestamp, in femtoseconds (default 1)
            
        Return
        ------
        real_time : int
            real time of counting for the current chunk, in femtoseconds
        live_time : int
            live time of counting for the current chunk, in femtoseconds
        real_time_unc : int
            real time standard uncertainty of counting for the current chunk, in femtoseconds
        live_time_unc : int
            live time standard uncertainty of counting for the current chunk, in femtoseconds
        """
        diffs = [item[1] - item[0] for item in self.gatetimes]
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
            array of int indicating number of counts for each channel with index as the channel number
        """
        energies = np.array([item for item in self.deadtimes], dtype=int)
        counts = pd.Series(energies).value_counts(sort=False, ascending=True)
        return np.array([counts.get(key, 0) for key in range(self.channels)], dtype=int)
  
    def _extend_deadtime(self):
        """extend the dead time in case of overlapping triggers
        
        Parameters
        ----------
        None
            
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
                    self.deadtimes[_idx] = (self.deadtimes[_idx][0], self.deadtimes[_edx][1], self.deadtimes[_idx][2], self.deadtimes[_idx][3])

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
        
    def _add_deadtime_to_trigger_selection(self, signal_sequence, dead_time):
        """extend the dead time for not rejected triggers
        
        Parameters
        ----------
        signal_sequence : list
            list of 3-items tuple for triggers (time, energy, flag)
        dead_time : int
            extended dead time to apply to gate times, in femtoseconds
            
        Return
        ------
        signal_sequence_with_extended_deadtime : list
            descr
        """
        #0x0         DEAD TIME EXTENSION
        #0x1         DEAD TIME EXTENSION
        #0x8         WHAT IS THIS? IGNORE IT (fake event)
        #0x10        DEAD TIME EXTENSION
        #0x20        DEAD TIME EXTENSION (trigger lost is occurred before this event)
        #0x40        DEAD TIME EXTENSION (N triggers have been lost)
        #0x80        DEAD TIME EXTENSION (the event is saturating inside the gate)
        #0x100       DEAD TIME EXTENSION
        #0x200       DEAD TIME EXTENSION (first event after a board busy condition)
        #0x400       DEAD TIME EXTENSION (the input is saturating)
        #0x800       DEAD TIME EXTENSION
        #0x1000      event not matched in the time correlation filter ?????
        #0x4000      DEAD TIME EXTENSION (event with fine time stamp)
        #0x8000      DEAD TIME EXTENSION (piled-up event)
        
        rejected_signals = (hex(0x8),)

        return [(row[0], row[0] + dead_time, row[1], row[2]) for row in signal_sequence if row[2] not in rejected_signals]
        
    def _manage_gatetime(self):
        """aggregate dead timed when closed gates superimpose
        
        Parameters
        ----------
        None
            
        Return
        ------
        None
        """
        
        check_condition = [first[1] > second[0] for first, second in zip(self.gatetimes[:-1], self.gatetimes[1:])] + [False]
        
        _idx = 0
        while True:
            try:
                if check_condition[_idx] == True:
                    #something is happening
                    _edx = _idx + 1
                    while check_condition[_edx] == True:
                        _edx += 1

                    #aggregation
                    self.gatetimes[_idx] = (self.gatetimes[_idx][0], self.gatetimes[_edx][1])#extended deadtime
                    #the second one
                    #self.gatetimes[_idx] = (self.gatetimes[_idx][0], self.gatetimes[_idx][1])
                    #consider the second as the closing gate trigger
                    #self.gatetimes[_idx] = (self.gatetimes[_idx][0], self.gatetimes[_edx][0])

                    #deletion
                    for _sweepidx in range(1, _edx - _idx + 1, 1):
                        self.gatetimes[_idx + _sweepidx] = None
                    _idx = _edx + 1

                else:
                    _idx += 1

            except IndexError:
                break

        #filter
        self.gatetimes = [row for row in self.gatetimes if row is not None]
        
    def _flag_selection(self, flag):
        """manage flags
        
        0x0         nothing (I guess it's good)
        0x1         dead-time is occurred before this event (only firmware < 128.64)
        0x2         time stamp roll-over
        0x4         time stamp reset from external
        0x8         fake event
        0x10        memory full is occurred before this event
        0x20        trigger lost is occurred before this event
        0x40        N triggers have been lost
        0x80        the event is saturating inside the gate
        0x100       1024 triggers have been counted
        0x200       first event after a board busy condition
        0x400       the input is saturating
        0x800       N triggers have been counted
        0x1000      event not matched in the time correlation filter
        0x4000      event with fine time stamp
        0x8000      piled-up event
        0x80000     identifies a fake event reporting a PLL lock loss
        0x100000    identifies a fake event reporting an over-temperature condition
        0x200000    identifies a fake event reporting an ADC shutdown
        
        Parameters
        ----------
        flag : str
            descr
            
        Return
        ------
        flag_accepted : bool
            descr
        """
        pass
        
    def _intersection(self, _previous_endpoint, _gate_close):
        """manage intersection between triggers and dead time
        
        Parameters
        ----------
        _previous_endpoint : str
            descr
        _gate_close : bool
            descr
            
        Return
        ------
        trigger : list
            list of energies (channels) of the trigger counts
        shuffled : list
            list of 2-item tuples indicating start and end periods of the dead time
        """
        
        off_scale = ()#'0x80', '0x400', '0x8000', '0xc00' '0x8800', '0x800', '0x880', '0x480')

        i1, i2 = 0, 0

        while True:
            try:
                if _previous_endpoint > self.deadtimes[i1][0]:
                    i1 += 1
                else:
                    break
            except IndexError:
                break

        while True:
            try:
                if _previous_endpoint > self.gatetimes[i2][0]:
                    i2 += 1
                else:
                    break
            except IndexError:
                break

        #i1 and i2 separetely
        if i1 > 0 and i2 == 0:
            endpoint = self.deadtimes[i1-1][1]
            self.deadtimes = self.deadtimes[i1:]
            if _previous_endpoint < endpoint:
                self.gatetimes = [(_previous_endpoint, endpoint)] + self.gatetimes
        elif i2 > 0 and i1 == 0 and _gate_close:
            #do not extend dead time
            self.gatetimes = self.gatetimes[i2:]
        elif i2 > 0 and i1 == 0 and not _gate_close:
            #extend dead time
            endpoint = self.gatetimes[i2-1][1]
            self.gatetimes = self.gatetimes[i2:]
            if _previous_endpoint < endpoint:
                self.gatetimes = [(_previous_endpoint, self.gatetimes[i2-1][1])] + self.gatetimes
        #else:
        elif i1 > 0 and i2 > 0:
            endpointT = self.deadtimes[i1-1][1]
            self.deadtimes = self.deadtimes[i1:]
            
            endpointG = self.gatetimes[i2-1][1]
            self.gatetimes = self.gatetimes[i2:]

            if _gate_close:
                endpointG = _previous_endpoint
                
            if endpointT > endpointG:
                endpoint = endpointT
            else:
                endpoint = endpointG
                
            try:
                _intersect = endpoint > self.gatetimes[0][0]
            except IndexError:
                _intersect = False
            
            if _previous_endpoint < endpoint and not _intersect:
                self.gatetimes = [(_previous_endpoint, endpoint)] + self.gatetimes
            elif _previous_endpoint < endpoint and _intersect:
                self.gatetimes[0] = (_previous_endpoint, self.gatetimes[0][1])

        shuffled = []
        trigger = []
        i1, i2 = 0, 0
        #shuffled list might be unordered but it's not an issue as only the sum of differences is needed from now on

        while True:
            try:
                aa = self.deadtimes[i1]
            except IndexError:
                shuffled += self.gatetimes[i2:]
                break
            try:
                bb = self.gatetimes[i2]
            except IndexError:
                trigger += [item[2] for item in self.deadtimes[i1:] if item[3] not in off_scale]
                shuffled += self.deadtimes[i1:]
                break

            if aa[0] < bb[0] and aa[1] <= bb[0]:
                #no intersection between aa and bb, aa trigger accepted
                if aa[3] not in off_scale:
                    trigger.append(aa[2])
                shuffled.append(aa)
                i1 += 1

            elif aa[0] < bb[0] and aa[1] <= bb[1]:
                #intersection between aa and bb, aa trigger accepted
                if aa[3] not in off_scale:
                    trigger.append(aa[2])
                shuffled.append((aa[0], bb[0]))
                i1 += 1

            elif aa[0] < bb[0] and aa[1] >= bb[1]:
                #intersection between aa and bb, aa trigger accepted
                if aa[3] not in off_scale:
                    trigger.append(aa[2])
                shuffled.append((aa[0], bb[0]))
                shuffled.append((bb[1], aa[1]))
                i1 += 1

            elif bb[0] <= aa[0] and bb[1] <= aa[0]:
                #no intersection between aa and bb
                shuffled.append(bb)
                i2 += 1

            elif bb[0] <= aa[0] and bb[1] >= aa[1]:
                #intersection between aa and bb
                i1 += 1

            elif bb[0] <= aa[0] and bb[1] <= aa[1]:
                #intersection between aa and bb
                shuffled.append((bb[1], aa[1]))
                i1 += 1

            else:
                #something failed
                raise NotImplementedError(f'unexpected condition encountered in loop\nTRIG: {aa[0]}, {aa[1]}\nGATE: {bb[0]}, {bb[1]}')

        return trigger, shuffled


class ElaborationSpectrum(GammaSpectrum):
    """Defines a spectrum class containing all the relevant information
    and that is suitable for peak evaluation (inherits from GammaSpectrum)
    
    Attributes
    ----------
    self.elaboration : pk_elaboration.Elaboration
        stores the elaboration of peaks
    self.elaboration_date : datetime ?
        descr
    self.elaborated_peak_list : list ?
        starting timestamp of the chunk
    self.problems : str ?
        descr
    
    Methods
    -------
    _open_file()
        retrieve information from spectrum file
    filename()
        returns the filename of the spectrum
    _autocalibration() ?
        descr
    _centroid_search()
        scan the spectrum to find peaks' centroids
    init_elaboration()
        initialize the elaboration object
    _save()
        save the object on file
    """
    def __init__(self, fullpath, calibration, K=3.5):
        """GammaSpectrum class constructor
        
        Parameters
        ----------
        fullpath : str (or Path)
            path of file .asc containing information about the spectrum
        calibration : Calibration
            calibration object
        K : float
            descr (default 3.5)
            
        Return
        ------
        None        
        """
        self.__spectrumpath = fullpath
        self.elaboration_date = None
        
        real_time, live_time, counts, real_time_unc, live_time_unc, start_acquisition_date, ext_deadtime = self._open_file()
        
        super().__init__(real_time=real_time, live_time=live_time, counts=counts, real_time_unc=real_time_unc, live_time_unc=live_time_unc, channels=len(counts), extended_dead_time=ext_deadtime)
        
        self.datetime = start_acquisition_date

        self.elaborated_peak_list = None
        self.problems = None
		#experimental
        self.elaboration = self.init_elaboration(calibration, K)
        
    @property
    def spectrumpath(self):
        """Set self.spectrumpath as a property
        """
        return self.__spectrumpath
        
    def _open_file(self):
        """Return the content of self.spectrumpath by calling openASCfile()
    
        Parameters
        ----------
        None
        
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
        return openASCfile(self.spectrumpath)
        
    def filename(self):
        """Return the filename of the spectrum without path
    
        Parameters
        ----------
        None
        
        Return
        ------
        filename : str
            filename of spectrum, without path
        """
        filename=str(self.spectrumpath)
        filename=str.split(filename,'/')[-1]
        if filename.lower().endswith(('.asc', '.chn')):
            return filename[:-4]
        return filename
        
    def _autocalibration(self, energies):
        #experimental
        _autocentroid_search(self, height=None, threshold=0.05, distance=4, min_prominence=2, wlen=10, rel_height=0.5, plateau_size=(1,2), E_min=None, E_max=None)
        
    def _centroid_search(self, looked_for_energies=None, tol=1.5, first_peaks=25):
        #experimental
        centroids = pk_elab._autocentroid_search(self.counts, first_peaks=first_peaks)
        if looked_for_energies is not None:
            data_check = np.array(sorted(looked_for_energies))
        
        acc_centroids = []
        
        for item in centroids:
            corrsp_energy = self.elaboration._get_energy(item)
            if looked_for_energies is not None:
                test_check = (data_check > corrsp_energy - tol) & (data_check < corrsp_energy + tol)
                if np.sum(test_check) > 0:
                    test = 'keep'
                else:
                    test = 'reject'
            else:
                test = 'no selection' 
            print(f'{item} -> {corrsp_energy:.2f} keV | {test}')
            if test != 'reject':
                acc_centroids.append(item)
        
        return acc_centroids
        
    def init_elaboration(self, calibration, K=3.5):
        """Define elaboration of spectrum
    
        Parameters
        ----------
        calibration : int
            calibration assigned for peak elaboration
        K : float
            number of FWHM to identify the width of a region (default 3.5)
        
        Return
        ------
        elaboration : pk_elab.Elaboration
            elaboration object
        """
        #just for test
        calibration = Calibration()

        return pk_elab.Elaboration(self, calibration=calibration, K=K)
        
    def _save(self):
        """save elaborated spectrum on disc as a binary file with pickle
        
        Parameters
        ----------
        None
            
        Return
        ------
        None
        """
        path, filename = os.path.split(self.spectrumpath)
        fname, extension = os.path.splitext(filename)
        with open(os.path.join(path, f'{fname}.spa'),'wb') as filesave:
            pickle.dump(self, filesave)


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
        uncertianty of real time in s
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

def open_digitized_spectrum(filename, chunksize=100000, channel_signal=0, channel_gate=2, extended_deadtime=100, spectrum_channels=16384, verbose=True, LFC=False):#maybe loss-free counting (LFC)
    """Return a time value (converted in s) from the corresponding dataframe line
    
    Parameters
    ----------
    filename : str (path)
        name of the gamma spectrum in list mode file
    chunksize : int
        number of rows to elaborate at each iteration; useful for huge files (default 100000)
    channel_signal : int
        descr (default 0)
    channel_gate : int
        descr (default 2)
    extended_deadtime : float
        time in us (default 100)
    spectrum_channels : int
        number of channels in the spectrum (default 16384)
    LFC : bool
        whether perform elaboration in loss-free counts method, useful for spectra with rapidly changing dead times (default False)
    
    Return
    ------
    spectrum : GammaSpectrum
        elaborated spectrum
    """

    with pd.read_csv(filename, sep=';', header=0, names=('BOARD', 'CHANNEL', 'TIMETAG', 'ENERGY', 'FLAGS'), chunksize=chunksize) as spectrum_file:
        
        #initialize the spectrum
        spectrum = GammaSpectrum(channels=spectrum_channels)
        spectrum.extended_dead_time = str(extended_deadtime)
        
        _previous_endpoint = 0
        _gate_close = False
        criticals = (hex(0x2), hex(0x4), hex(0x80000), hex(0x100000), hex(0x200000))

        for chunk in spectrum_file:
            
            #pre-check for errors
            _CRITICAL_ERROR = chunk['FLAGS'].isin(criticals)
            if np.sum(_CRITICAL_ERROR) > 0:
                #Warning
                idx = chunk.index[chunk[_CRITICAL_ERROR]]
                print(f'CRITICAL ERROR AT LINE {idx[0]}')
                if LFC:
                    spectrum.counts = np.round(spectrum.counts).astype(int)
                return spectrum
            
            _filter_signal = chunk['CHANNEL'] == channel_signal
            
            # no need to filter the double trigger for each signal as it will be neglected with the extended dead time if an extended dead time > 2 microsecond is selected (5 to be more confident), 
            # however there will be a small increase of dead time in the order of a couple microseconds

            #collect and elaborate chunks of information
            local_ch = SpectrumChunk([(trigger_time, energy, flag) for trigger_time, energy, flag in zip(chunk[_filter_signal]['TIMETAG'], chunk[_filter_signal]['ENERGY'], chunk[_filter_signal]['FLAGS'])], extended_deadtime, [(trigger_time, flag) for trigger_time, flag in zip(chunk[~_filter_signal]['TIMETAG'],chunk[~_filter_signal]['FLAGS'])], _previous_endpoint, _gate_close, channels=spectrum.channels, LFC=LFC)

            _previous_endpoint = local_ch.end
            _gate_close = local_ch.gate_close
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
    return spectrum

def spectrum_multielab(filename, extended_deadtimes, n_channels, chunksize=250000, folder='elaboration', verbose=True, LFC=False):
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
    folder : str (or Path)
        name of the folder where the output is saved (default 'elaboration')
    verbose : bool
        descr (default True)
    LFC : bool
        whether perform elaboration in loss-free counts method, useful for spectra with rapidly changing dead times (default False)
    
    Return
    ------
    None
    """

    for EXTENDED_DEADTIME in extended_deadtimes:
        spectrum = open_digitized_spectrum(filename, chunksize=chunksize, extended_deadtime=EXTENDED_DEADTIME, spectrum_channels=n_channels, verbose=verbose, LFC=LFC)
        
        if not os.path.isdir(folder):
            os.makedirs(folder)
        path, file_name = os.path.split(filename)
        fname, extension = os.path.splitext(file_name)
        
        spectrum._save(os.path.join(folder, fname))
        
        print(f'EXTENDED DEAD TIME {EXTENDED_DEADTIME} μs COMPLETED!')
        
def multi_spectrum_elab(filenames, extended_deadtimes, n_channels, chunksize=250000, folder='elaboration', verbose=True, LFC=False):#CHECK THIS OUT
    """Elaboration of the same list mode spectrum with multiple extended dead times
    
    Parameters
    ----------
    filenames : list
        name of the gamma spectrum in list mode file
    extended_deadtimes : list
        list of floats indicating extended dead times in μs to feed the elaboration
    n_channels : int
        number of channels in the spectrum
    chunksize : int
        number of rows to elaborate at each iteration; useful for huge files (default 250000)
    folder : str (or Path)
        name of the folder where the output is saved (default 'elaboration')
    verbose : bool
        descr (default True)
    LFC : bool
        whether perform elaboration in loss-free counts method, useful for spectra with rapidly changing dead times (default False)
    
    Return
    ------
    None
    """

    for _file in filenames:
        spectrum_multielab(extended_deadtimes)
        print(f'FILE {_file} COMPLETED!')

def display_results(folder='test'):
    #folder = 'test'
    flist = os.listdir(folder)
    edt_set = sorted(set([int(fname.split('_')[1]) for fname in flist]))
    
    fig = plt.figure(figsize=(16,8), layout="constrained")

    gs = GridSpec(3, 4, figure=fig)
    ax_sp = fig.add_subplot(gs[0:2, :])
    ax_live = fig.add_subplot(gs[2, 0])
    ax_counts = fig.add_subplot(gs[2, 1], sharex=ax_live)
    ax_CR = fig.add_subplot(gs[2, 2], sharex=ax_live)
    ax_DT = fig.add_subplot(gs[2, 3], sharex=ax_live)

    pos, lab = [], []
    results = {}
    for nn, item in enumerate(edt_set):
        spct = ElaborationSpectrum(os.path.join(folder, f'EDT_{item}_spectrum.asc'), Calibration(), K=10)
        info1 = pd.read_csv(os.path.join(folder, f'EDT_{item}_info1.csv'))
        info2 = pd.read_csv(os.path.join(folder, f'EDT_{item}_info2.csv'))
        info3 = pd.read_csv(os.path.join(folder, f'EDT_{item}_info3.csv'))
        
        ax_sp.plot(np.arange(len(spct.counts)), spct.counts, linestyle='-', label=rf'{item} $\mu$s', alpha=0.5)
        ax_live.bar(nn, np.max(np.cumsum(info1.live_time)), width=1, alpha=0.5, align='edge', label=rf'{item} $\mu$s')
        ax_counts.bar(nn, np.max(np.cumsum(info1.sum_counts)), width=1, alpha=0.5, align='edge', label=rf'{item} $\mu$s')
        ax_CR.bar(nn, np.max(np.cumsum(info1.sum_counts)) / np.max(np.cumsum(info1.live_time)), width=1, alpha=0.5, align='edge', label=rf'{item} $\mu$s')
        ax_DT.bar(nn, (np.max(np.cumsum(info1.real_time)) / np.max(np.cumsum(info1.live_time)) - 1)*100, width=1, alpha=0.5, align='edge', label=rf'{item} $\mu$s')
        
        pos.append(nn+0.5)
        lab.append(rf'{item} $\mu$s')
        
        #proper peak elaboration
        centrs = spct._centroid_search([80.57, 184.41, 280.46, 410.95], tol=1)
        
        for _centr in centrs:
            spct.elaboration._add_region([_centr])
            
        peaks = {}
        for region in spct.elaboration.regions:
            x_plot = np.arange(region.low_limit, region.high_limit, 1)
            y_plot = spct.counts[x_plot]
            
            region.elaboration_fit(spct.elaboration.spectrum_profile, spct.elaboration.cal_fwhm, peak_opts={'bound_peak':2, 'bound_fwhm':3})
            peaks[(region.low_limit, region.high_limit)] = {'fit':region.fit, 'x_plot':x_plot, 'y_plot':y_plot}
        results[item] = peaks

    ax_sp.set_ylabel('counts / 1')
    ax_live.set_ylabel('live time / s')
    ax_counts.set_ylabel('total counts / 1')
    ax_CR.set_ylabel('count rate / 1')
    ax_DT.set_ylabel('dead time / %')
    ax_sp.legend(loc='best', ncols=5, fontsize='small')
    ax_sp.set_xlim(0, len(spct.counts))
    ax_sp.set_ylim(0.9, None)
    ax_sp.set_yscale('log')
    ax_live.set_xlim(0, len(edt_set))
    ax_live.set_xticks(pos, lab, rotation=90)
    ax_counts.set_xticks(pos, lab, rotation=90)
    ax_CR.set_xticks(pos, lab, rotation=90)
    ax_DT.set_xticks(pos, lab, rotation=90)
    fig.tight_layout()
    plt.show()


    fig = plt.figure(figsize=(16,8), layout="constrained")

    gs = GridSpec(3, 3, figure=fig)
    ax_reg1 = fig.add_subplot(gs[0, 0])
    ax_reg2 = fig.add_subplot(gs[0, 1])
    ax_reg3 = fig.add_subplot(gs[0, 2])
    
    ax_res1 = fig.add_subplot(gs[1, 0], sharex=ax_reg1)
    ax_res2 = fig.add_subplot(gs[1, 1], sharex=ax_reg2)
    ax_res3 = fig.add_subplot(gs[1, 2], sharex=ax_reg3)
    
    ax_area1 = fig.add_subplot(gs[2, 0])
    ax_area2 = fig.add_subplot(gs[2, 1])
    ax_area3 = fig.add_subplot(gs[2, 2])
    
    axreg = (ax_reg1, ax_reg2, ax_reg3)
    axres = (ax_res1, ax_res2, ax_res3)
    axarea = (ax_area1, ax_area2, ax_area3)
    poss = [[], [], []]
    labs = [[], [], []]
    for ii, _key in enumerate(sorted(results.keys())):
        _value = results[_key]
        for nn, s_key in enumerate(sorted(_value.keys(), key=lambda x : x[0])):
            x_plot = _value[s_key]['x_plot']
            y_plot = _value[s_key]['y_plot']
            _fit = _value[s_key]['fit']
            x_fit = np.linspace(s_key[0], s_key[1], 500)
            if _fit is not None:
                y_fit = _fit._draw_total(x_fit)
                axreg[nn].plot(x_fit, y_fit, label=rf'{_key} $\mu$s')
                y_res = (_fit._draw_total(x_plot) / y_plot - 1) * 100
                axres[nn].plot(x_plot, y_res, marker='o', markersize=1.5, linestyle='-')
                
            axreg[nn].set_xlim(np.min(x_plot), np.max(x_plot))
            axreg[nn].set_ylim(0.9, None)
            if axreg[nn].get_title() is None or axreg[nn].get_title() == '':
                axreg[nn].set_title(f'{spct.elaboration._get_energy(_fit.centroids[0]):.1f} keV')
            
            axarea[nn].errorbar(ii, _fit.pks[0], yerr = 2 * _fit.ur_pks[0] * _fit.pks[0], marker='o', markersize=2, elinewidth=0.75)
            poss[nn].append(ii) 
            labs[nn].append(rf'{_key} $\mu$s ({_fit})')
    
    for ax in axreg:
        ax.set_ylabel('counts / 1')
        ax.set_xticklabels([])
    
    for ax in axres:
        ax.axhline(y=3, xmin=0, xmax=1, linestyle='-', linewidth=0.75, color='#950606')
        ax.axhline(y=-3, xmin=0, xmax=1, linestyle='-', linewidth=0.75, color='#950606')
        ax.axhline(y=2, xmin=0, xmax=1, linestyle='-.', linewidth=0.5, color='#950606')
        ax.axhline(y=-2, xmin=0, xmax=1, linestyle='-.', linewidth=0.5, color='#950606')
        ax.set_ylabel('residuals / 1')
        
    for ax, pos, lab in zip(axarea, poss, labs):
        ax.set_xticks(pos, lab, rotation=90)
        ax.set_ylabel('net area / 1')
    
    plt.show()

filename = 'gamma/SDataR_Ho-166m_IRA_1_UG_2026-01-08.csv'

CHUNKSIZE = 50000
EXTENDED_DEADTIMES = (5, 10, 25, 50, 75, 100, 125, 150, 175, 200)
EXTENDED_DEADTIMES = (5,) # 5 μs is basically almost no dead time
N_CHANNELS = 16384

if __name__ == '__main__':
    #pass
    spectrum_multielab(filename, EXTENDED_DEADTIMES, N_CHANNELS, chunksize=CHUNKSIZE, LFC=True)
    
    """

    ccdf = ElaborationSpectrum('10_100_spectrum.asc', Calibration(), K=10)
    css = ccdf._centroid_search([184])
    print(css)
    ccdf.elaboration._add_region(css)
    ccdf.elaboration.regions[0].elaboration_fit(ccdf.counts, Calibration().fwhm_params)
    
    print(ccdf.elaboration.regions[0].fit)
    
    if ccdf.elaboration.regions[0].fit is not None:
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(6,7), sharex=True, height_ratios=[2,1])
        xx = np.arange(ccdf.elaboration.regions[0].low_limit, ccdf.elaboration.regions[0].high_limit)
        ax1.bar(xx, ccdf.counts[xx], width=1, align='edge', alpha=0.7)
        
        xfit = np.linspace(ccdf.elaboration.regions[0].low_limit, ccdf.elaboration.regions[0].high_limit, 500)
        yfit = ccdf.elaboration.regions[0].fit._draw_total(xfit)
        yBKG = ccdf.elaboration.regions[0].fit._draw_background(xfit)
        
        ax1.plot(xfit, yfit, linestyle='-', marker='', color='r')
        ax1.plot(xfit, yBKG, linestyle='-', marker='', color='k')
        
        yres = ccdf.elaboration.regions[0].fit._draw_total(xx)/ccdf.counts[xx] - 1
        
        ax2.plot(xx, yres * 100, linestyle='-', marker='o', markersize=2.5, color='r')
        
        ax1.set_xlim(ccdf.elaboration.regions[0].low_limit, ccdf.elaboration.regions[0].high_limit)
        ax1.set_ylim(1E3, None)
        ax1.set_yscale('log')
        
        fig.tight_layout()
        plt.show()"""