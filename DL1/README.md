# Preprocessing

We did not spent a lot of time with the preprocess.  
Even though something intresting could have been done with the different sizes of the images.  
We basically used the tools provided by Keras DataImageGenerator, that rezised the images too the same given input size, and generated some data augmentation by doing some transformations to the images.  
We also separated the dataset into 2 folders: for validation and for training. The Validation was 10% of the whole dataset.  



# Begining

We first started with the model given as example for the mnist dataset. We adapted it in order to take our dataset input size and classification types. We even let the images stay in grayscale.

[Accuracy](plots/first_cnn_acc.pdf)
[Loss](plots/first_cnn_loss.pdf)

We can see in the plots that the model did not learn.
The model was not complex enough.

## Model
conv- 8 - 4x4  
maxPool - 2x2  
conv- 16- 2x2  
maxPool - 2x2  
Flatten  
Dense - 8  
Dense - 5  

# Complexing the model

We tried to complex the model by adding a Convolutional layer, changing the kernels sizes and augmenting the filters.

[Accuracy](plots/2_cnn_acc.pdf)
[Loss](plots/2_cnn_loss.pdf)

We see that the model even though it got a little better, still did not learn.

## Model
conv- 16 - 16x16  
conv- 32 - 8x8  
maxPool - 2x2  
conv- 64- 4x4  
maxPool - 2x2  
Flatten  
Dense - 512  
Dense - 5  


# A model from kaggle

We then checked a model from kaggle to start with.


**TODO**

[Accuracy](plots/3_cnn_acc.pdf)
[Loss](plots/3_cnn_loss.pdf)


# More convolutional layer

We tried to add a convolutional layer with the same kernel same but a more filters to follow the structure. For this we had to also resize the pictures.

[Accuracy](plots/4_cnn_acc.pdf)
[Loss](plots/4_cnn_loss.pdf)


**TODO**
