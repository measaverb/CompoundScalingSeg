# CompoundScalingSeg

There are many ways to improve the performance of segmentation. In this work, I suggest improving the performance by compound scaling. Compound Scaling is a method that EfficientNet used to improve a base network made by Neural Architecture Search. By increasing the size of the width, depth of a network, and resolution of the input image can make image classification better.

In this repository,  I adapt the compound scaling technique at nerve segmentation.

This experiment comprises three ways of experiments. First, scaling the size of the layer width. This process increases the number of channels. It expects to get better by widening.

Second, scaling the size of the input image. Scaling the size of the input image also expects to get better.

Lastly, by scaling these two methods stated above, even better performance and an efficient network for segmentation are possible.
