import config_test as config
import numpy as np

def active_microphones(mode, mics_used):
    # depending on the chosen mode, the correct microphone indexes are calculated
    # and stored in the list active_mics
    #       mode = 1: alla mikrofoner, 
    #       mode = 2: varannan
    #       mode = 3: var tredje
    #       mode = 4: var fjärde
    #       (visualisera array setup med att sätta plot_setup = 1 i config.py)
    rows = np.arange(0, config.ROWS, mode)                              # number of rows in array
    columns = np.arange(0, config.COLUMNS*config.ACTIVE_ARRAYS, mode)   # number of columns in array
    elements = config.N_MICROPHONES          # total number of elements in the array configuration
    mics = np.linspace(0, elements-1, elements)           # vector holding all microphone indexes for all active arrays
    arr_elem = config.ROWS*config.COLUMNS                               # number of elements in one array

    # microphone indexes for one array, in a matrix
    microphones = np.linspace(0, config.ROWS*config.COLUMNS-1,config.ROWS*config.COLUMNS).reshape((config.ROWS, config.COLUMNS))

    # for each additional array, stack a matrix of the microphone indexes of that array
    for a in range(config.ACTIVE_ARRAYS-1):
        a += 1
        array = mics[0+a*arr_elem : arr_elem+a*arr_elem].reshape((config.ROWS, config.COLUMNS))
        microphones = np.hstack((microphones, array))

    if mics_used == 'all':
        unused_mics = []        # TA BORT SEDAN OCH AVKOMMNENTERA RADERNA OVAN!!!

    if mics_used == 'only good':
        #take out the active microphones from the microphones matrix, save in list active_mics
        try:
            unused_mics = np.load('unused_mics.npy')
        except:
            unused_mics = []

    active_mics = []
    for r in rows:
        for c in columns:
            mic = microphones[r,c]
            if mic not in unused_mics:
                #continue
                active_mics.append(int(mic))

    # sort the list such that the mic indexes are in ascending order
    active_mics = np.sort(active_mics)
    return active_mics

