# Keras DRAGAN

Implementation of DRAGAN(Deep Regret Analytic Generative Adversarial Networks) with keras.

Paper: [here](https://arxiv.org/abs/1705.07215)

Reference: [here](https://github.com/tjwei/GANotebooks/blob/master/dragan-keras.ipynb)


![DRAGAN_Alg](./image/DRAGAN_alg.png)

# Dataset
I got images from Twitter,Pinterest, and [Getchu](http://www.getchu.com/). Total is about 16000 images



# Demo
I build [website](https://girls-gan.herokuapp.com/index.html#/) with Tensorflow.js (recommend chorme)



# Result 

### DRAGAN 


imagesize: 64x64
batchsize: 128


![DRAGAN](./image/result.png) 

### DRAGAN + Residual
 
imagesize: 64x64
batchsize: 128


![DRAGAN4](./residual/residual128.png) 
 
### DRAGAN + RDN

imagesize: 128x128
batchsize: 32


![DRAGAN3](./RDN/result.png)


### DRAGAN + EDSR

imagesize: 128x128
batchsize: 32


![DRAGAN2](./EDSR/result.png)



### latent
![latent](./image/latent.jpg)


## Environment
- OS: Windows 10
- CPU: Intel(R) Core(TM)i7-8700
- GPU: NVIDIA GTX1060 6GB
- RAM: 16GB
