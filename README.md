# scale_detection_1d
Training simple CNNs on the Scale-Detection task. Compute their sensitivities to input transformations.

#create_dataset
Creates the Scale-Detection dataset in 1 dimension. 
It is possible to choose the characteristic length \xi and the image size, and whether Task 1 and Task 2.

#convolutional_nets
Simple CNNs used to learn Scale-Detection task, with ReLU non linearity.
Parameters: filter size, stride, number of channels, depth.

#ScaleDetection_SGD
Training simple CNNs on the Scale-Detection task.
input parameters:
characteristic scale 
image size 
gap
number of layers
learning rate
number channels
magnitude active pixels in the dataset
ridge for weight decay
p for the weight decay p-norm

#diffeo_cluster_mean
