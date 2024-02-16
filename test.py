
import numpy as np
import matplotlib.pyplot as plt

lambdas = [0, 0.001, 0.01, 0.1, 1]
w_weight = [0.0030511910031241008, 0.0026967538156860075, 0.0013385389839289845, 0.00034802300215133855, 4.615691759652412e-05]
for i in range(5):
    w_weight[i]*=28*785
plt.scatter(lambdas, w_weight)
plt.xlabel("Lambdas")
plt.ylabel("Weight length")
plt.show()