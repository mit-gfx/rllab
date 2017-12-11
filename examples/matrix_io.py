import numpy as np
import struct

def ReadMatrixFromFile(file_name):
    with open(file_name, mode='rb') as f:
        content = f.read()
    row, col = struct.unpack('ii', content[:8])
    data = struct.unpack('d' * ((len(content) - 8) // 8), content[8:])
    m = np.array(data)
    m = np.reshape(m, (row, col), 'F')
    return m

def WriteMatrixToFile(file_name, matrix_data):
    pass
