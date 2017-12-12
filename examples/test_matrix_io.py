import numpy as np
import matrix_io as m

A = np.random.rand(2, 3)
print(A)
m.WriteMatrixToFile('A', A)
A2 = m.ReadMatrixFromFile('A')
print(A2)
