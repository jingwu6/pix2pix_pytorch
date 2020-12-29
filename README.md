# pix2pix_pytorch
Pix2pix based on Wasserstein GAN




## Code structure:

The code consists of 6 python files, which are:


Main.py: It is the main file that takes in all the arguments including all the hyperparameters.

Test.py: It takes all the test set to give the outputs of the model.

Train.py: This is the most important part of the code that contains a class which defines the whole training process of pixel2pixel. 

Utils.py: It contains several functions used in training. They include set neural network not update, sample buffer, learning rate scheduler and initialization function for neural networks.

Models: It contains two kinds of model for training: resnet and Unet based on resnet18

datasets:It helps read the datasets.


## Run the following code in terminal to train and test:

python main.py 


## Reference:  

1.Original paper: Image-to-Image Translation with Conditional Adversarial Networks  
2.Wasserstein GAN: https://arxiv.org/pdf/1701.07875.pdf  
3.A clean and readable Pytorch implementation of CycleGAN: https://github.com/aitorzip/PyTorch-CycleGAN
