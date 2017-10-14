"""
YOU THOUGHT YOU HAD DESTROYED ME
BUT I HAVE RETURNED ONCE MORE
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np

with open("3_layer.pkl",'rb') as saveFile:
    NN_3_layer = pickle.load(saveFile)

with open("4_layer.pkl",'rb') as saveFile:
    NN_4_layer = pickle.load(saveFile)

with open("5_layer.pkl",'rb') as saveFile:
    NN_5_layer = pickle.load(saveFile)

plt.figure()
plt.plot(range(1000), NN_3_layer, label="3 layer")
plt.plot(range(1000), NN_4_layer, label="4 layer")
plt.plot(range(1000), NN_5_layer, label="5 layer")
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.title('Test Error')
plt.legend()
plt.savefig('p1b4_compare.png')
plt.show()
