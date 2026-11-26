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

#settings
CHUNKSIZE = 50000         # size of every chunk in lines
EXTENDED_DEADTIMES = (5,) # list of extended dead times in µs
N_CHANNELS = 16384        # number of channel of the resulting spectrum
LossFreeCounts = False    # whether to perform a Loss-Free Counts correction at every chunk
verbose = True            # whether to be verbose during elaboration of each chunk
full_output = True        # whether to return an additional csv output with comprehensive information of every chunk

spectrum_object = spectra.open_digitized_spectrumPHA(filename, EXTENDED_DEADTIMES, N_CHANNELS, chunksize=CHUNKSIZE, verbose=verbose, LFC=LossFreeCounts, folder='C:/.../gamma/elaboration', full_output=full_output)

#OR with GUI_support
kw = GUI_support.get_options()

spectrum_object = spectra.open_digitized_spectrumPHA(**kw)


#multiple elaboration are possible, explore methods within spectra
