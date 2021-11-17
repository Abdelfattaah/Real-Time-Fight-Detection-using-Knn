# Real-Time-Fight-Detection-using-Knn
### - This Code is written in "C++" and based on the "openpose" library, so you need to setup the openpose enviroment before you run this code
#### - You can find the repo of the openpose [Here](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
----------------------------------
### [Here](https://docs.google.com/presentation/d/1TjjxTctxQaoWI28SeY8l0qwlTq-AiWXv/edit?usp=sharing&ouid=116579051541464863394&rtpof=true&sd=true) is the presentation file that i used for this project in case someone needs it

### This is a demo video of the final project

[![Watch the video](https://img.youtube.com/vi/LOOCKCUbIk0/maxresdefault.jpg)](https://youtu.be/LOOCKCUbIk0)
----------------------------------
## System Overview:

![figure 1](https://user-images.githubusercontent.com/40593273/142265402-703d7897-a42b-4fea-98d7-fd5b23584815.png)<br />
###### (Figure 1)

- This system is composed of 3 stages, The first stage is dataset loading and processing, the first step in this stage is loading the data, and the second step is applying the openpose library on the dataset to detect the joints from the human bodies, then the third step is to apply mathematical equations on the coordinates of the joints we have got to calculate the angle of each body joint (figure 2), and then the last step is to store the angels we got from the dataset (figure 1)

![figure 2](https://user-images.githubusercontent.com/40593273/142264853-943f7f9a-4ada-49b5-9359-1b1dbd3a530e.png)<br /> 
###### (Figure 2)

- The Second stage is reading and processing the input video frame by frame and applying the same steps in the previous stage as shown in (figure 1).

- The third stage is applying the knn algorithm on the stored data to decide whether the frame of the input video containing a violence or not (figure 3).

![figure 3](https://user-images.githubusercontent.com/40593273/142265732-f8209355-a6c2-4c7b-b502-c1b33a00b9f5.png)<br />
###### (Figure 3)

## DataSet
- Our dataset contains 187 image, The images consist violence and non-violence examples downloaded from the internet and screenshots has taken from movies, it’s 110 image of violence and 77 image of non-violence.

## Evaluation
-	Accuracy: 
about 65%

-	Space: 
The data set is About 17 MB

-	Performance: 
About 10 seconds to feed the model in the first time. </br > 
- Regarding the time, the frame needs < 1 sec to be processed, and most of the time needed is for the dataset feeding.	
