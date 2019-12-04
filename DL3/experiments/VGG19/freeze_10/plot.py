import matplotlib.pyplot as plt
import pickle
import os
import sys
import glob

def load_test(file_name):
	f = open(file_name, "rb")
	obj = pickle.load(f)
	return obj

path = glob.glob("*.history")[0]

data = load_test(path)


eps = len(data["loss"])



## Plot different layers for 64 nodes training
epochs = range(1, eps+1)
plt.title("Loss")
# plt.ylabel("Amount")
plt.xlabel("Epochs")

plt.plot(epochs, data["loss"][0:eps], color="blue", label="training")
plt.plot(epochs, data["val_loss"][0:eps], color="red", label="validation")
plt.legend()
plt.savefig("loss.png")

plt.clf()
plt.xlabel("Epochs")
plt.title("Accuracy")
plt.plot(epochs, data["acc"][0:eps], color="green", label="training")
plt.plot(epochs, data["val_acc"][0:eps], color="orange", label="validation")
plt.legend()
plt.savefig("accuracy.png")

