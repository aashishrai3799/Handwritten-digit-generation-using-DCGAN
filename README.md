# Handwritten-digit-generation-using-DCGAN
Generating handwritten digits using Deep Convolutional Generative Adversarial Network(DCGAN).

A Generative Adversarial Network (GAN) is a class of machine learning systems invented by Ian Goodfellow[1] in 2014. 
Two neural networks contest with each other in a game (in the sense of game theory, often but not always in the form 
of a zero-sum game). Given a training set, this technique learns to generate new data with the same statistics as the 
training set. For example, a GAN trained on photographs can generate new photographs that look at least superficially 
authentic to human observers, having many realistic characteristics.

This code uses the MNIST dataset to train itself and afterwards generates the similar images.
The MNIST database (Modified National Institute of Standards and Technology database) is a large database of handwritten 
digits that is commonly used for training various image processing systems.
The MNIST database contains 60,000 training images and 10,000 testing images.

A Generative Adversarial Network has two main Neural Networks.

*   Generator
*   Discriminator

# Generator

The Generator takes a random vector(Noise), and generates a 32x32 image. The generator tries to resambles the generated image
with the input MNIST dataset images.

# Discriminator

The discriminator distinguish between the real images and the generated fake images. Its input is a 32x32 image and output is
its probability of being real.

#### Enviromnent: Python 3.7

# Required Libraries:
*   tensorflow__1.14
*   numpy______1.16.4
*   matplotlib___3.0.3
*   scipy________1.2.0
*   os
*   time

# Working of Network

Run the model for 1,50,000 epochs or more to obtain the optimum result. You can increase/decrease the learning rates and 
check the variation in the results. The provided values may not be the most accurate, but have worked well so far.

Do check the sample_image folder to see how model learns.

#### This model is typically designed for GPUs, running it on a CPU may be quite strenous.
#### I ran it on Tesla T4 GPU and it took around 1 minute to train 1000 epochs.

## References:

*   GAN original paper by Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley,
    Sherjil Ozair, Aaron Courville, Yoshua Bengio
*   en.wikipedia.org
