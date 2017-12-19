import numpy as np
import struct
import sys
from datetime import datetime 


def ReadMatrixFromFile(file_name):
    startTime= datetime.now() 


    with open(file_name, mode='rb') as f:
        content = f.read()
    row, col = struct.unpack('ii', content[:8])
    data = struct.unpack('d' * ((len(content) - 8) // 8), content[8:])
    m = np.array(data)
    m = np.reshape(m, (row, col), 'F')


    timeElapsed=datetime.now()-startTime 

    #print('ReadMatrix: Time elpased (hh:mm:ss.ms) {}'.format(timeElapsed))
    return m

def WriteMatrixToFile(file_name, matrix):
    startTime= datetime.now() 

    # First, get the row and col of the array.
    shape = matrix.shape
    if len(shape) == 1:
        matrix = matrix[:,np.newaxis]
    row, col= matrix.shape
    row_bytes = row.to_bytes(4, sys.byteorder, signed=True)
    col_bytes = col.to_bytes(4, sys.byteorder, signed=True)
    # Second, get the raw data.
    data_bytes = matrix.astype(np.float64, 'F').tobytes('F')
    # Finally, write the binary file.
    f = open(file_name, 'w+b')
    content = b"".join([row_bytes, col_bytes, data_bytes])
    f.write(content)
    f.close()

    timeElapsed=datetime.now()-startTime 

    #print('WriteMatrix: Time elpased (hh:mm:ss.ms) {}'.format(timeElapsed))
