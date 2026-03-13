"""
Module to convert list mode files to csv readable from spectra module

at present only ORTEC/MAESTRO files are implemented, the conversion is started with:
read_ORTEC_ListMode('string of path to ORTEC list mode file to convert')

I would like to acknowledge Robin Peter, the author of gamma-bits (https://github.com/robin-peter/gamma-bits), from which a few fucntions were remodeled to be included here

"""


import os
import datetime
import numpy as np
import pandas as pd
import struct
from bitstring import BitArray


def _read_MAESTRO_header(header):
    """Decode information from header of a ORTEC list file.

    Bytes interpretation gathered from ORTEC document 'list-mode-file formats.pdf' found at:
    https://www.ortec-online.com/-/media/ametekortec/manuals/l/list-mode-file-formats.pdf?la=en&revision=0b78b32a-9ee7-4243-b4ac-a8adb886381d

    Parameters
    ----------
    header : bytes
        256-long bytes header of a ORTEC list file


    Return
    ------
    spectrum_info : dict
        dictionary containing all relevant information extracted from the header
    """

    REFERENCE_TIME_OLE = datetime.datetime(1899, 12, 30, 0, 0, 0)

    h_format = int.from_bytes(header[:4], 'little', signed=True)
    data_style = int.from_bytes(header[4:8], 'little', signed=True)
    acquisition_start_OLE = 86400 * struct.unpack('<d', header[8:16])[0]

    s = acquisition_start_OLE // 1
    us = round(acquisition_start_OLE % 1, 3) * 1e6

    acquisition_start = REFERENCE_TIME_OLE + datetime.timedelta(seconds=s, microseconds=us)
    detector = header[16:96].decode()
    mcb = header[96:105].decode()
    detector_serial = header[105:121].decode()
    description = header[121:201].decode()

    E_control = header[201:202].decode()
    E_unit = header[202:206].decode()
    E0 = struct.unpack('<f', header[206:210])[0]
    E1 = struct.unpack('<f', header[210:214])[0]
    E2 = struct.unpack('<f', header[214:218])[0]

    if E_control not in ('0', ''):
        E_params = np.array([E0, E1, E2])
    else:
        E_params = np.array([np.nan, np.nan, np.nan])

    FWHM_control = header[218:219].decode()
    FWHM0 = struct.unpack('<f', header[219:223])[0]
    FWHM1 = struct.unpack('<f', header[223:227])[0]
    FWHM2 = struct.unpack('<f', header[227:231])[0]

    if FWHM_control not in ('0', ''):
        FWHM_params = np.array([FWHM0, FWHM1, FWHM2])
    else:
        FWHM_params = np.array([np.nan, np.nan, np.nan])

    conversion_gain = int.from_bytes(header[231:235], 'little', signed=True)
    detector_ID = int.from_bytes(header[235:239], 'little', signed=True)
    real_time = struct.unpack('<f', header[239:243])[0]
    live_time = struct.unpack('<f', header[243:247])[0]
    #char = header[247:255].decode() #unused bytes

    return {'header_format':h_format, 'data_style':data_style, 'acquisition_start':acquisition_start, 'detector':detector, 'mcb':mcb, 'detector_serial':detector_serial, 'description':description, 'E_unit':E_unit, 'E_control':E_control, 'E_params':E_params, 'FWHM_control':FWHM_control, 'FWHM_params':FWHM_params, 'conversion_gain':conversion_gain, 'detector_ID':detector_ID, 'real_time':real_time, 'live_time':live_time}

def _read_ADC(word, real_time_reference):
    """Decode information about timestamp and enegy of the event.


    This function is an adapted verision of an analogous function from gamma-bits/process_listmode.py written by robin-peter found at:
    https://github.com/robin-peter/gamma-bits

    Parameters
    ----------
    word : str
        string representation of binary data concerning event timetag and energy
    real_time_reference : int
        value of the current real time tick in fs


    Return
    ------
    channel : int
        channel proportional to energy of the event
    time : int
        time in fs corresponding to the timetag of the event
    """

    FINE_TIME_SAMPLE = 200000

    readableADCWord = word[-6:] + word[16:24]
    channel = int(readableADCWord, 2)

    # combine fine timestamp with most recent coarse timestamp
    readableFineTimeWord = word[8:16] + word[:8]
    timestamp = FINE_TIME_SAMPLE * int(readableFineTimeWord, 2) + real_time_reference

    return int(channel), int(timestamp)

def _read_time(word):
    """Decode information about time ticks.


    This function is an adapted verision of an analogous function from gamma-bits/process_listmode.py written by robin-peter found at:
    https://github.com/robin-peter/gamma-bits

    Parameters
    ----------
    word : str
        string representation of binary data concerning event timetag and energy
    real_time_reference : int
        value of the current real time tick in fs


    Return
    ------
    time : int
        time in fs corresponding to the completion of a counting time in 10 ms steps
    """

    readableRTWord = word[-6:] + word[16:24] + word[8:16] + word[:8]
    time = int(readableRTWord, 2) * 1E10
    return int(time)

def accumulate_triggers(data, columns):
    """Organize spectrum event information into a pd.DataFrame in order to be readable by other functions from spectra module.


    Parameters
    ----------
    data : list
        list of 2-item tuples containing data for energy end timestamp of events
    columns : list (tuple)
        name of the columns of the pd.DataFrame


    Return
    ------
    df : pd.DataFrame
        pd.DataFrame with data organized in columns
    """

    df = pd.DataFrame(data=data)
    df.rename(columns={0:columns[3], 1:columns[2]}, inplace=True)
    #other columns
    df[columns[0]] = 0
    df[columns[1]] = 0
    df[columns[4]] = '0x0'
    return df

def write_trigger_onfile(df, outDataFile, columns):
    """Write pd.DataFrame on file.


    Parameters
    ----------
    df : pd.DataFrame
        pd.DataFrame returned from accumulate_triggers() function
    outDataFile : str (Path)
        filepath for the saved file
    columns : list (tuple)
        name of the columns of the pd.DataFrame


    Return
    ------
    None
    """

    df.to_csv(outDataFile, sep=';', mode='a', index=False, header=False, columns=columns)

def _read_MAESTRO_datatype2(dataarray, outDataFile, chunk=50000):
    """Read binary data of ORTEC file.


    Parameters
    ----------
    dataarray : str
        string representation of binary file
    outDataFile : str (Path)
        filepath for the saved file
    chunk : int
        number of lines to write on file each time (default 50000)


    Return
    ------
    None
    """

    real_time_reference = 0
    chunk = int(chunk)
    data = []
    columns = ['BOARD', 'CHANNEL', 'TIMETAG', 'ENERGY', 'FLAGS']
    pd.DataFrame(data=[], columns=columns).to_csv(outDataFile, sep=';', header=True, index=False, mode='w', columns=columns)

    for word in dataarray:
        encoderByte = word[-8:]
        if encoderByte[:2] == '11':  # ADC word
            channel, timestamp = _read_ADC(word, real_time_reference)
            # write trigger on output file
            data.append((channel, timestamp))
            if len(data) >= chunk:
                #flush
                df = accumulate_triggers(data, columns)
                data = []
                write_trigger_onfile(df, outDataFile, columns)
        elif encoderByte[:2] == '10':  # RT word
            real_time_reference = _read_time(word)
        elif encoderByte[:2] == '01':  # LT word
            pass#_read_time(word)
        elif encoderByte == '00000000':  # HDW
            pass#handleHDWWord(word)
        elif encoderByte == '00000001':  # C1
            pass#handleC1Word(word)
        elif encoderByte == '00000010':  # C2
            pass#handleC2Word(word)
        elif encoderByte == '00000011':  # C3
            pass#handleC3Word(word)
        else:
            pass

    if len(data) > 0:
        df = accumulate_triggers(data, columns)
        data = []
        write_trigger_onfile(df, outDataFile, columns)


def read_ORTEC_ListMode(inListModeFile):
    """Read ORTEC list mode file (.Lis) and convert it to csv.


    Parameters
    ----------
    inListModeFile : str (Path)
        path of an ORTEC list mode file to convert to csv


    Return
    ------
    None
    """

    WORD_BIT_LENGTH = 32
    HEADER_BYTE_LENGTH = 256

    root, _ = os.path.splitext(inListModeFile)

    with open(inListModeFile, mode='rb') as f:
        fileContent = f.read()

    Header = fileContent[:HEADER_BYTE_LENGTH]
    ListModeData = fileContent[HEADER_BYTE_LENGTH:]
    ListModeBitString = BitArray(bytes=ListModeData).bin
    ListModeWordArray = [ListModeBitString[i:i+WORD_BIT_LENGTH] for i in range(0, len(ListModeBitString), WORD_BIT_LENGTH)]

    spectrum_info = _read_MAESTRO_header(Header)
    try:
        str_date = spectrum_info['acquisition_start'].strftime("%Y-%m-%d %H_%M_%S")
        _date = f'_AS_{str_date}'
    except (KeyError, TypeError, ValueError):
        _date = ''
    outDataFile = f'{root}{_date}.csv'

    #data interpretation
    if spectrum_info['data_style'] == 1:
        pass
    elif spectrum_info['data_style'] == 2:
        _read_MAESTRO_datatype2(ListModeWordArray, outDataFile)
    elif spectrum_info['data_style'] == 3:
        pass
    elif spectrum_info['data_style'] == 4:
        pass
