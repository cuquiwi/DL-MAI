import matplotlib.pyplot as plt
import pickle
import os
import sys
import glob

def load_test(file_name):
	f = open(file_name, "rb")
	obj = pickle.load(f)
	return obj

paths = glob.glob("*.history")

data = [] 
for path in paths:
	data.append([path[:-8], load_test(path)])


typ = "val_acc"


## Plot different layers for 64 nodes training
plt.title(typ)

plt.xlabel("Epochs")

for dat in data:
	name = dat[0]
	eps = len(dat[1][typ])
	epochs = range(1, eps+1)
	plt.plot(epochs, dat[1][typ][0:eps], label=name)
	#plt.plot(epochs, dat[1]["val_loss"][0:eps], label=name)

plt.legend()
plt.savefig(typ +".png")
"""
plt.clf()
plt.title("Accuracy")
plt.plot(epochs, data["acc"][0:eps], color="green", label="training")
plt.plot(epochs, data["val_acc"][0:eps], color="orange", label="validation")
plt.legend()
plt.savefig("accuracy.png")

"""