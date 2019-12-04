import matplotlib.pyplot as plt
import pickle
import os

def load_test(file_name):
	f = open(file_name, "rb+")
	obj = pickle.load(f)
	return obj

path = "../trainHistoryDict"

data = load_test(path)

eps = 50



## Plot different layers for 64 nodes training
epochs = range(1, eps+1)
plt.title("MobileNet training")
# plt.ylabel("Amount")
plt.xlabel("Epochs")

plt.plot(epochs, data["loss"][0:eps], color="blue", label="loss")
plt.plot(epochs, data["acc"][0:eps], color="red", label="accuracy")
plt.legend()
plt.savefig("../../../plots/MobileNet_training.png")

plt.clf()
plt.plot(epochs, data["val_loss"][0:eps], color="green", label="val_loss")
plt.plot(epochs, data["val_acc"][0:eps], color="orange", label="val_accuracy")
plt.legend()
plt.savefig("../../../plots/MobileNet_validation.png")

