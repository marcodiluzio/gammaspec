## Quick start

Parsing of the csv LIST MODE files is already functional  

from opening a python shell in the folder gammaspec:  
```python
(Virtual Environment) PS C:\...\gammaspec\src\gammaspec> python
```

it's possible to open and parse csv LIST MODE files by executing:  

```python
import spectra
import GUI_support #for GUI help during setting selection
import peaks       #for peak elaboration

#settings for list mode files acquired with CAEN digitizers
CHUNKSIZE = 250000        # size of every chunk in lines
EXTENDED_DEADTIMES = 5    # float of extended dead time to apply in µs
N_CHANNELS = 16384        # number of channel of the resulting spectrum
LossFreeCounts = False    # whether to perform a Loss-Free Counts correction at every chunk
verbose = True            # whether to be verbose during elaboration of each chunk

spectrum_object = spectra.open_digitized_spectrumPHA(filename, EXTENDED_DEADTIMES, N_CHANNELS, chunksize=CHUNKSIZE, verbose=verbose, LFC=LossFreeCounts, folder='C:/.../gamma/elaboration', full_output=full_output)

#OR with GUI_support
kw = GUI_support.get_options()

spectrum_object = spectra.open_digitized_spectrumPHA(**kw)


#settings for list mode files acquired with FPGA card
CHUNKSIZE = 250000        # size of every chunk in lines
EXTENDED_DEADTIMES = 5    # float of extended dead time to apply in µs
N_CHANNELS = 16384        # number of channel of the resulting spectrum
ADC_conv = 32             # conversion factor for charges to channels
LossFreeCounts = False    # whether to perform a Loss-Free Counts correction at every chunk
verbose = True            # whether to be verbose during elaboration of each chunk

spectrum_object = spectra.open_digitized_spectrumFPGA(filename, EXTENDED_DEADTIMES, N_CHANNELS, chunksize=CHUNKSIZE, ADC_conv=ADC_conv, verbose=verbose, LFC=LossFreeCounts, folder='C:/.../gamma/elaboration', full_output=full_output)


#multiple elaboration are possible, explore methods within spectra
