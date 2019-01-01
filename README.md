# Keras DRAGAN

Implementation of DRAGAN(Deep Regret Analytic Generative Adversarial Networks) with keras.

Paper: [here](https://arxiv.org/abs/1705.07215)

Reference: [here](https://github.com/tjwei/GANotebooks/blob/master/dragan-keras.ipynb)

# Dataset
I got images from Twitter,Pinterest, and [Getchu](http://www.getchu.com/). Total is about 16000 images

学習する部分を減らすために、データセットの背景は白で統一した方が綺麗な画像ができるようです


# Demo
I build [website](https://girlsgan.herokuapp.com/index.html#/) with Tensorflow.js (recommend chorme)

<!--
# Example

You need ".npy" file in advance. Then,
```bash
$ python DRAGAN.py
``` -->

# Result

![DRAGAN](./result.png)

### latent
![latent](./latent.jpg)

- Input size: 64x64
- Batch size: 128
- This is 1000epochs( It takes 9 hours to get this result with GTX 1060)

やっぱりDeconvolutionよりPixel Shufflerをつかったほうが綺麗だし学習速いですね

## Environment
- OS: Windows 10
- CPU: Intel(R) Core(TM)i7-8700
- GPU: NVIDIA GTX1060 6GB
- RAM: 8GB
