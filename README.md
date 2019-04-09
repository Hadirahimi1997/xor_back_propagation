# xor_back_propagation
Solving XOR problem with Back Propagation Algorithm

### Introduction

XOR or Exclusive OR is a classic problem in Artificial Neural Network Research.
An XOR function takes two binary inputs (0 or 1) & returns True if both inputs are different & False if both inputs are same.

| Input 1 | Input 2 | Output  |
|:-:|:-:|:-:|
| 0 | 0 | 0 |
| 1 | 1 | 0 |
| 1 | 0 | 1 |
| 0 | 1 | 1 |

On the surface, XOR appears to be a very simple problem, however, Minksy and Papert (1969) showed that this was a big problem for neural network architectures of the 1960s, known as perceptrons.
A limitation of this architecture is that it is only capable of separating data points with a single line. This is unfortunate because the XOR inputs are not linearly separable. This is particularly visible if you plot the XOR input values to a graph. 
As shown in the figure, there is no way to separate the 1 and 0 predictions with a single classification line.

![image](https://cdn-images-1.medium.com/max/800/0*qdRb80zUpJPtrbRD.)

### Solution

The backpropagation algorithm begins by comparing the actual value output by the forward propagation process to the expected value and then moves backward through the network, slightly adjusting each of the weights in a direction that reduces the size of the error by a small degree. Both forward and back propagation are re-run thousands of times on each input combination until the network can accurately predict the expected output of the possible inputs using forward propagation.

### Model

![image](https://user-images.githubusercontent.com/46073809/55764569-07099580-5a21-11e9-9b84-4db7543f4632.png)

**Inputs**

![image](https://user-images.githubusercontent.com/46073809/55764661-6e274a00-5a21-11e9-8cda-270b3c36a868.png)

2 hidden neurons are used, each takes two inputs with different weights.
After each forward pass, the error is back propogated.
I have used sigmoid as the activation function at the hidden layer.

![image](https://user-images.githubusercontent.com/46073809/55764747-bfcfd480-5a21-11e9-9619-43e516fc8250.png)

### Model

#### Classification **WITHOUT** gaussian noise

![image](https://user-images.githubusercontent.com/46073809/55764818-ff96bc00-5a21-11e9-9786-60752ab39b76.png)

**Observation :** To classify, I generated 1000 x 2 random floats in range -2 to 2. Using weights from trained model, I classified each input & plotted it on 2-D space. Out of 1000, 731 points were classified as “+1” and 269 points were classified as “-1”
It is clearly seen, classification region is not a single line, rather the 2-D region is separated by “-1” class.
I did the same for classifications with Gaussian noise.

#### Classification **WITH** gaussian noise

In real applications, we almost never work with data without noise. Now instead of using the above points generate Gaussian random noise centered on
these locations.

![image](https://user-images.githubusercontent.com/46073809/55764980-7df35e00-5a22-11e9-89e1-da6264a02f16.png)

##### σ = 0.5
![image](https://user-images.githubusercontent.com/46073809/55765040-bbf08200-5a22-11e9-9680-6c046d1badcb.png)

**Observation :** We see a shift in classification regions. Here, classification is still not separated by a line

##### σ = 1.0
![image](https://user-images.githubusercontent.com/46073809/55765054-cb6fcb00-5a22-11e9-8cb3-fc9a0c4b7d0a.png)

**Observation :** We observe classifications being divided into two distinct regions.

##### σ = 2.0
![image](https://user-images.githubusercontent.com/46073809/55765062-d4609c80-5a22-11e9-8a7c-6774d40ec8e7.png)

**Observation :** We see a shift in classification regions (compared to of σ = 1). Here also, we observe two distinct regions of classification.

