import matplotlib.pyplot as plt
import pickle
import os

def load_test(file_name):
	f = open(file_name, "rb+")
	obj = pickle.load(f)
	return obj

dirs = next(os.walk('.'))[1]
dirs.remove("data")
dirs.remove("plots")
try: 
	dirs.remove("__pycache__")
except:
	pass

paths = {}
datas = {}

for dir in dirs:
	path = os.getcwd() + "/" + dir + "/trainHistoryDict"
	data = load_test(path)
	datas[dir] = data






## Plot different layers for 64 nodes training
epochs = range(1, 201)
plt.title("GRU 64 nodes per layer training")
# plt.ylabel("Amount")
plt.xlabel("Epochs")

plt.plot(epochs, datas["1_layer_64_nodes"]["loss"], color="blue", label="loss 1 layer")
#plt.plot(epochs, datas["1_layer_64_nodes"]["acc"], color="blue", linestyle="--", label="acc 1 layer")

plt.plot(epochs, datas["2_layer_64_nodes"]["loss"], color="red", label="loss 2 layers")
#plt.plot(epochs, datas["1_layer_64_nodes"]["acc"], color="red", linestyle="--", label="acc 2 layers")

plt.plot(epochs, datas["3_layer_64_nodes"]["loss"], color="green", label="loss 3 layers")
#plt.plot(epochs, datas["1_layer_64_nodes"]["acc"], color="green", linestyle="--", label="acc 3 layers")

plt.plot(epochs, datas["4_layer_64_nodes"]["loss"], color="orange", label="loss 4 layers")
#plt.plot(epochs, datas["1_layer_64_nodes"]["acc"], color="orange", linestyle="--", label="acc 4 layers")
plt.legend()
plt.savefig("./plots/64_training.png")


## Plot different layers for 64 nodes validation
plt.clf()
plt.title("GRU 64 nodes per layer validation")
# plt.ylabel("Amount")
plt.xlabel("Epochs")

plt.plot(epochs, datas["1_layer_64_nodes"]["val_loss"], color="blue", label="val_loss 1 layer")
#plt.plot(epochs, datas["1_layer_64_nodes"]["val_acc"], color="blue", linestyle="--", label="val_acc 1 layer")

plt.plot(epochs, datas["2_layer_64_nodes"]["val_loss"], color="red", label="val_loss 2 layers")
#plt.plot(epochs, datas["1_layer_64_nodes"]["val_acc"], color="red", linestyle="--", label="val_acc 2 layers")

plt.plot(epochs, datas["3_layer_64_nodes"]["val_loss"], color="green", label="val_loss 3 layers")
#plt.plot(epochs, datas["1_layer_64_nodes"]["val_acc"], color="green", linestyle="--", label="val_acc 3 layers")

plt.plot(epochs, datas["4_layer_64_nodes"]["val_loss"], color="orange", label="val_loss 4 layers")
#plt.plot(epochs, datas["1_layer_64_nodes"]["val_acc"], color="orange", linestyle="--", label="val_acc 4 layers")
plt.legend()
plt.savefig("./plots/64_validation.png")


## Plot different layers for 128 nodes training
plt.clf()
plt.title("GRU 128 nodes per layer training")
# plt.ylabel("Amount")
plt.xlabel("Epochs")

plt.plot(epochs, datas["1_layer_128_nodes"]["loss"], color="blue", label="loss 1 layer")
#plt.plot(epochs, datas["1_layer_128_nodes"]["acc"], color="blue", linestyle="--", label="acc 1 layer")

plt.plot(epochs, datas["2_layer_128_nodes"]["loss"], color="red", label="loss 2 layers")
#plt.plot(epochs, datas["1_layer_128_nodes"]["acc"], color="red", linestyle="--", label="acc 2 layers")

plt.plot(epochs, datas["3_layer_128_nodes"]["loss"], color="green", label="loss 3 layers")
#plt.plot(epochs, datas["1_layer_128_nodes"]["acc"], color="green", linestyle="--", label="acc 3 layers")

plt.plot(epochs, datas["4_layer_128_nodes"]["loss"], color="orange", label="loss 4 layers")
#plt.plot(epochs, datas["1_layer_128_nodes"]["acc"], color="orange", linestyle="--", label="acc 4 layers")
plt.legend()
plt.savefig("./plots/128_training.png")


## Plot different layers for 128 nodes validation
plt.clf()
plt.title("GRU 128 nodes per layer validation")
# plt.ylabel("Amount")
plt.xlabel("Epochs")

plt.plot(epochs, datas["1_layer_128_nodes"]["val_loss"], color="blue", label="val_loss 1 layer")
#plt.plot(epochs, datas["1_layer_128_nodes"]["val_acc"], color="blue", linestyle="--", label="val_acc 1 layer")

plt.plot(epochs, datas["2_layer_128_nodes"]["val_loss"], color="red", label="val_loss 2 layers")
#plt.plot(epochs, datas["1_layer_128_nodes"]["val_acc"], color="red", linestyle="--", label="val_acc 2 layers")

plt.plot(epochs, datas["3_layer_128_nodes"]["val_loss"], color="green", label="val_loss 3 layers")
#plt.plot(epochs, datas["1_layer_128_nodes"]["val_acc"], color="green", linestyle="--", label="val_acc 3 layers")

plt.plot(epochs, datas["4_layer_128_nodes"]["val_loss"], color="orange", label="val_loss 4 layers")
#plt.plot(epochs, datas["1_layer_128_nodes"]["val_acc"], color="orange", linestyle="--", label="val_acc 4 layers")
plt.legend()
plt.savefig("./plots/128_validation.png")


## Plot training vs validation (64 and 128 together), 1 layer
plt.clf()
plt.title("GRU 1 layer")
# plt.ylabel("Amount")
plt.xlabel("Epochs")
plt.plot(epochs, datas["1_layer_64_nodes"]["loss"], color="blue", linestyle="--", label="loss 64 nodes")
plt.plot(epochs, datas["1_layer_64_nodes"]["val_loss"], color="blue", label="val_loss 64 nodes")
plt.plot(epochs, datas["1_layer_128_nodes"]["loss"], color="green", linestyle="--", label="loss 128_nodes")
plt.plot(epochs, datas["1_layer_128_nodes"]["val_loss"], color="green", label="val_loss 128 nodes")
plt.legend()
plt.savefig("./plots/1_comp.png")


## Plot training vs validation (64 and 128 together), 2 layers
plt.clf()
plt.title("GRU 2 layers")
# plt.ylabel("Amount")
plt.xlabel("Epochs")
plt.plot(epochs, datas["2_layer_64_nodes"]["loss"], color="blue", linestyle="--", label="loss 64 nodes")
plt.plot(epochs, datas["2_layer_64_nodes"]["val_loss"], color="blue", label="val_loss 64 nodes")
plt.plot(epochs, datas["2_layer_128_nodes"]["loss"], color="green", linestyle="--", label="loss 128_nodes")
plt.plot(epochs, datas["2_layer_128_nodes"]["val_loss"], color="green", label="val_loss 128 nodes")
plt.legend()
plt.savefig("./plots/2_comp.png")


## Plot training vs validation (64 and 128 together), 2 layers
plt.clf()
plt.title("GRU 3 layers")
# plt.ylabel("Amount")
plt.xlabel("Epochs")
plt.plot(epochs, datas["3_layer_64_nodes"]["loss"], color="blue", linestyle="--", label="loss 64 nodes")
plt.plot(epochs, datas["3_layer_64_nodes"]["val_loss"], color="blue", label="val_loss 64 nodes")
plt.plot(epochs, datas["3_layer_128_nodes"]["loss"], color="green", linestyle="--", label="loss 128_nodes")
plt.plot(epochs, datas["3_layer_128_nodes"]["val_loss"], color="green", label="val_loss 128 nodes")
plt.legend()
plt.savefig("./plots/3_comp.png")


## Plot training vs validation (64 and 128 together), 2 layers
plt.clf()
plt.title("GRU 4 layers")
# plt.ylabel("Amount")
plt.xlabel("Epochs")
plt.plot(epochs, datas["4_layer_64_nodes"]["loss"], color="blue", linestyle="--", label="loss 64 nodes")
plt.plot(epochs, datas["4_layer_64_nodes"]["val_loss"], color="blue", label="val_loss 64 nodes")
plt.plot(epochs, datas["4_layer_128_nodes"]["loss"], color="green", linestyle="--", label="loss 128_nodes")
plt.plot(epochs, datas["4_layer_128_nodes"]["val_loss"], color="green", label="val_loss 128 nodes")
plt.legend()
plt.savefig("./plots/4_comp.png")