## Quick start

Parsing of the csv LIST MODE files is already functional  

from opening a python shell script in the folder gammaspec:  
```python
(Virtual Environment) PS C:\...\gammaspec\src\gammaspec> python
```

it's possible to open and parse csv LIST MODE files by executing:  

```python
import spectra

#settings
CHUNKSIZE = 50000         # size of every chunk in lines
EXTENDED_DEADTIMES = (5,) # list of extended dead times in µs
N_CHANNELS = 16384        # number of channel of the resulting spectrum
LossFreeCounts = False    # whether to perform a Loss-Free Counts correction at every chunk
full_output = True        # whether to return an additional csv output with comprehensive information of every chunk

#elaboration on a single file
filename = 'C:/.../gamma/SDataR_Ho-166m_IRA_1_UG_2026-01-08.csv'
spectra.spectrum_multielab(filename, EXTENDED_DEADTIMES, N_CHANNELS, chunksize=CHUNKSIZE, verbose=True, LFC=False, folder='C:/.../gamma/elaboration')

#multiple files at once
data_folder = 'C:/.../gamma'
filenames = spectra.get_spectra_filenames(data_folder)
spectra.multi_spectrum_elab(filenames, EXTENDED_DEADTIMES, N_CHANNELS, chunksize=CHUNKSIZE, verbose=True, LFC=LossFreeCounts, folder=os.path.join(data_folder, 'elaboration'), full_output=full_output)
```

these functions return an elaborated spectrum in asc format containing the counts array and information about counting times
