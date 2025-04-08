import struct
import numpy as np 

def read_binary(fpath: str) -> np.ndarray:
 
    """
    Read a binary .tim file from the CSPB.ML dataset into a numpy array.
 
    https://cyclostationary.blog/2023/09/25/cspb-ml-2018r2-correcting-an-rng-flaw-in-cspb-ml-2018/

    An implementation of the read_binary MATLAB function cited below.
 
    https://cyclostationary.blog/wp-content/uploads/2015/09/read_binary.doc

    """

    with open(fpath, 'rb') as f: 

        data_type = struct.unpack('i',f.read(4))[0] 
        num_samples = struct.unpack('i',f.read(4))[0]

        #no complex numbers (not tested)
        if data_type == 1:
            data=np.frombuffer(f.read(), count=num_samples, dtype=np.float32)

        elif data_type == 2:
            data = np.frombuffer(f.read(), count=num_samples*2, dtype=np.float32) 
            data = data[0::2]+(data[1::2])*1j
        
    return data           




  
