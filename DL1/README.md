# Preprocessing

We did not spent a lot of time with the preprocess.  
Even though something intresting could have been done with the different sizes of the images.  
We basically used the tools provided by Keras DataImageGenerator, that rezised the images too the same given input size, and generated some data augmentation by doing some transformations to the images.  
We also separated the dataset into 2 folders: for validation and for training. The Validation was 10% of the whole dataset.  



# Beginning

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
This is a far more complex model. It follows also some notions that we've seen on the course,
like a fixed small kernel, a growing pyramidal structure, and some dropouts and normalizations.


[Accuracy](plots/3_cnn_acc.pdf)
[Loss](plots/3_cnn_loss.pdf)

This model actually learns something and has a pretty good accuracy without overfitting.  
We will try to modify this model to try to find an improvement,
 and then see the effects that might cause each modification.

## Model
conv - 64 - 3x3  
conv - 64 - 3x3  
maxPool - 2x2  
BatchNorm  
Dropout - 0.2  
conv - 128 - 3x3  
conv - 128 - 3x3  
maxPool - 2x2  
BatchNorm  
Dropout - 0.2  
conv - 256 - 3x3  
conv - 256 - 3x3  
maxPool - 2x2  
BatchNorm  
Dropout - 0.2  
conv - 512 - 3x3  
conv - 512 - 3x3  
maxPool - 2x2  
BatchNorm  
Dropout - 0.2  
Flatten  
Dense - 1000  
Dropout - 0.2  
Dense - 512  
Dropout - 0.2  
Dense - 5  


# More convolutional layer

We tried to add a convolutional layer with the same kernel same but a more filters to follow the structure. For this we had to also resize the pictures.

[Accuracy](plots/4_cnn_acc.pdf)
[Loss](plots/4_cnn_loss.pdf)

The model took more time to train, it's more complex, but the learning did not improve, so we keep the less complex model. 


## Model
conv - 64 - 3x3  
conv - 64 - 3x3  
maxPool - 2x2  
BatchNorm  
Dropout - 0.2  
conv - 128 - 3x3  
conv - 128 - 3x3  
maxPool - 2x2  
BatchNorm  
Dropout - 0.2  
conv - 256 - 3x3  
conv - 256 - 3x3  
maxPool - 2x2  
BatchNorm  
Dropout - 0.2  
conv - 512 - 3x3  
conv - 512 - 3x3  
maxPool - 2x2  
BatchNorm  
Dropout - 0.2  
conv - 1024 - 3x3  
conv - 1024 - 3x3  
maxPool - 2x2  
BatchNorm  
Flatten  
Dense - 1000  
Dropout - 0.2  
Dense - 512  
Dropout - 0.2  
Dense - 5  

# Model without BatchNormalization

What would happen if we removed the BatchNormalization?


[Accuracy](plots/5_cnn_acc.pdf)
[Loss](plots/5_cnn_loss.pdf)

The learning that we could see previously, in other models, is no more. 
This model does not learn at all, like the first models.
I guess this is due to some weights explosions or vanishment.
There is a weird phenomen that happens also, is that the validation is better than the training. We could not found an explanation for that.

## Model
conv - 64 - 3x3  
conv - 64 - 3x3  
maxPool - 2x2  
Dropout - 0.2  
conv - 128 - 3x3  
conv - 128 - 3x3  
maxPool - 2x2  
Dropout - 0.2  
conv - 256 - 3x3  
conv - 256 - 3x3  
maxPool - 2x2  
Dropout - 0.2  
conv - 512 - 3x3  
conv - 512 - 3x3  
maxPool - 2x2  
Dropout - 0.2  
Flatten  
Dense - 1000  
Dropout - 0.2  
Dense - 512  
Dropout - 0.2  
Dense - 5  

# Model without Dropouts

[Ioffe and Szegedy](https://arxiv.org/pdf/1502.03167v3.pdf) claimed that with Batch Normalization we can, in some cases, eliminate the need of Dropout. So we tried to completely eliminate the Dropout layers.

[Accuracy](plots/nodrop_cnn_acc.pdf)
[Loss](plots/nodrop_cnn_loss.pdf)

And indeed the results where pretty good. Since those results are pretty good and the model is less complex we will try to improve this model.

## Model
conv - 64 - 3x3  
conv - 64 - 3x3  
maxPool - 2x2  
BatchNorm  
conv - 128 - 3x3  
conv - 128 - 3x3  
maxPool - 2x2  
BatchNorm  
conv - 256 - 3x3  
conv - 256 - 3x3  
maxPool - 2x2  
BatchNorm  
conv - 512 - 3x3  
conv - 512 - 3x3  
maxPool - 2x2  
BatchNorm  
Flatten  
Dense - 1000  
Dense - 512  
Dense - 5  

# Model with adding/removing a hidden FNN layer

We try to add/remove an hidden layer of 512 nodes in the full connected part of the model.

## 512 Addition

[Accuracy](plots/cnn_Hidden512_add_acc.pdf)
[Loss](plots/cnn_Hidden512_add_loss.pdf)

conv - 64 - 3x3  
conv - 64 - 3x3  
maxPool - 2x2  
BatchNorm  
Dropout - 0.2  
conv - 128 - 3x3  
conv - 128 - 3x3  
maxPool - 2x2  
BatchNorm  
Dropout - 0.2  
conv - 256 - 3x3  
conv - 256 - 3x3  
maxPool - 2x2  
BatchNorm  
Dropout - 0.2  
conv - 512 - 3x3  
conv - 512 - 3x3  
maxPool - 2x2  
BatchNorm  
Dropout - 0.2  
Flatten  
Dense - 1000  
Dropout - 0.2  
Dense - 512  
Dropout - 0.2  
Dense - 512  
Dropout - 0.2  
Dense - 5  



## 512 Removal

[Accuracy](plots/cnn_Hidden512_rem_acc.pdf)
[Loss](plots/cnn_Hidden512_rem_loss.pdf)

conv - 64 - 3x3  
conv - 64 - 3x3  
maxPool - 2x2  
BatchNorm  
Dropout - 0.2  
conv - 128 - 3x3  
conv - 128 - 3x3  
maxPool - 2x2  
BatchNorm  
Dropout - 0.2  
conv - 256 - 3x3  
conv - 256 - 3x3  
maxPool - 2x2  
BatchNorm  
Dropout - 0.2  
conv - 512 - 3x3  
conv - 512 - 3x3  
maxPool - 2x2  
BatchNorm  
Dropout - 0.2  
Flatten  
Dense - 1000  
Dropout - 0.2  
Dense - 5  



# Models with different dropout rates

### 0.05

[Accuracy](plots/cnn_dropout_005_acc.pdf)
[Loss](plots/cnn_dropout_005_loss.pdf)

### 0.08

[Accuracy](plots/cnn_dropout_008_acc.pdf)
[Loss](plots/cnn_dropout_008_loss.pdf)



We can see that a decrease in dropout causes the training and validation accuracy of the model to be low (around 0.6).

# Augmenting the epochs

Up until now we have used a maximum of 100 epoch, what happens if we increase the epochs? 
The model will still learn? Will it overfit?

[Accuracy](plots/200epoch_cnn_acc.pdf)
[Loss](plots/200epoch_cnn_loss.pdf)

As expected the model did not improve much and it started to overfit.

