## Project Description

I got annoyed that my Apple Face ID couldn't recognize my face when I was wearing a mask outside, so I decided to build a model that could. Granted, I am aware of the difficulties with creating a robust system that works for everyone's face so I'm not bashing Apple for this. I actually think that they did a good thing to not release a system that isn't reliable when one is wearing face masks. 

Moreover, I primarily used OpenCV to process the dataset of images and TensorFlow for actually training and testing the models. I considered a lightweight Convolutional Neural Network structure called MobileNet as my Deep Learning model. I then tailored the model for binary classification (Kaveh or Not Kaveh) by using a Sigmoid function to configure my output. After about 10 epochs, I managed to acheive a ~97% validation accuracy on the validation folds. Furthermore, after saving my model to disk, I developed a pipeline script that takes as input a picture of a face with a mask and determines if the person wearing the mask is me (Kaveh). My intent is to use this script as part of a pipeline in a bigger project.

The following are some examples of inputting previously unseen images into the model:

**Kaveh test 1**

![](/Examples/Kaveh%201.png)

**Kaveh test 2**

![](/Examples/Kaveh%202.png)

**Not Kaveh test 1**

![](/Examples/Not%20Kaveh%201.png)

**Not Kaveh test 2**

![](/Examples/Not%20Kaveh%202.png)

Please first take a look at the Jupyter Notebook, and then scripts, it will be much more helpful!


## How to Build the Project:

1) I took lots of pictures of my face with a mask on and combined with the a dataset of other people's faces with masks (dataset mentioned in the credits). Unforunately, I can't upload the images of my face that was used for training, so technically you can't use the `Train_Model.py` to train a classifier to identify my face. However, you can very well use the script to train a classifier that would recognize your own face when you are wearing a mask. The required changes are actually kind of easy to do, all you have to do is change some parameters and upload pictures of your own face with a mask on for the model to train on! I have left `Train_Model.py` for references.

2) Once the model files are saved to disk, you can run the pipeline script `Detect_Kaveh_With_Mask.py` and feed as input the relative directory path of an image. The result of the classification is outputted to the console. I have commented out a portion of the code that saves the same picture but tagged with 'Kaveh' or 'Not Kaveh'.


## Credits

TensorFlow MobileNet: https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNet

Masked Images Obtained From: https://github.com/prajnasb/observations/tree/master/experiements/data
