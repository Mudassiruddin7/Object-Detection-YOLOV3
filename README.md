![YOLO](https://64.media.tumblr.com/221b974c7efa9e08a0e7c4d50df082b1/tumblr_nd2unh2szY1td670bo1_500.gif)

# _**Detecting custom objects in an image using YOLO V3**_
YOLO is the state of the art and industry used algorithm for object detection these days. This does the classification/detection by creating grids in the whole image and achieves high accuracy at a very good speed and in real time. We use a pre-trained model which has been trained on the COCO dataset and performs object detection which a very high degree of accuracy. 

# _**Base Paper**_
+ https://arxiv.org/abs/1804.02767

# _**Algorithm Description**_
YOLO is an abbreviation word for “You look only once”, where this algorithm has shaken most of the developers all around the world in the past 5 years, due to its immense ability to quickly learn and remember any object we have trained or labeled it. There is a unique method to work with Yolo models where we require the images to be labeled firstly using specific tools available online or installing it in your local machine. (Labeling software’s links are provided below). The labeling process might take a long time depending upon on your dataset. The images are labeled according to the specification made by the user, whether we'd like to detect multiple objects in an image are single object, depending on it, the text file is created with following parameters,
• The class of the objects, starting from 0,1, 2.
• The coordinates of the bounding box drawn.

![YOLO](https://miro.medium.com/v2/resize:fit:1400/1*ZbmrsQJW-Lp72C5KoTnzUg.jpeg)

**Reference**

+ https://machinelearningknowledge.ai/introduction-to-yolov5-object-detection-with-tutorial/
+ https://pytorch.org/hub/ultralytics_yolov5
+ https://github.com/ultralytics/yolov5

# _To have your own custom object detection, your data needs to be in the following format:_

**Sample Label File:**

The Label file consists of following items.
• The class of the object
• Top left coordinate of the bounding box
• Top right coordinate of the bounding box
• Bottom left coordinate of the bounding box
• Bottom right coordinate of the bounding box


# _**How to Execute?**_
So, before execution we have some pre-requisites that we need to download or install i.e., anaconda environment, python and a code editor.
**Anaconda**: Anaconda is like a package of libraries and offers a great deal of information which allows a data engineer to create multiple environments and install required libraries easy and neat.

**Download link:**

![Anaconda](https://1.bp.blogspot.com/-UJ1Ws2zZ9V4/TtMbG2ynJiI/AAAAAAAABbM/m6t2kuEhKdY/s1600/The-biggest-anaconda-snake-3.jpg)

https://www.anaconda.com/

**Python**: Python is a most popular interpreter programming language, which is used in almost every field. Its syntax is very similar to English language and even children and learning it nowadays, due to its readability and easy syntax and large community of users to help you whenever you face any issues.

**Download link:**

![Python](https://i0.wp.com/reptileworldfacts.com/wp-content/uploads/2019/05/male-blonde-super-tiger-reticulated-python.jpg?resize=351%2C351&ssl=1)

https://www.python.org/downloads/

**Code editor**: Code editor is like a notepad for a programming language which allows user to write, run and execute program which we have written. Along with these some code editors also allows us to debug, which usually allows users to execute the code line by line and allows them to see where and how to solve the errors. But I personally feel visual code is very good to work with any programming language and makes a great deal of attachment with user.

**Download links:**

![Vs code](https://schwabencode.com/contents/logos/VS2019-Badge.png) ![Pycharm](https://i0.wp.com/scracked.com/wp-content/uploads/2020/01/PyCharm-2019.3.4-Crack.png?fit=200%2C200&ssl=1)

+ https://code.visualstudio.com/Download, 
+ https://www.jetbrains.com/pycharm/download/#section=windows

# _**How to create a new environment and configure jupyter notebook with it.**_
Let us define an environment and why we need different environments. An environment is a collection of libraries that are required to run our project. When we already have an environment with the necessary libraries, why do we need a new environment?
To avoid version mismatches, we create a new environment for each project. For example, in your previous project, you used "tf env" with tensorflow 2.4 and keras 2.4, but in your current project, you must use tensorflow 2.6 and keras 2.6. If you continue your project in the "tf env" environment, there will be a version mismatch and you will need to update tensorflow and keras, but this will cause problems with the previous project's execution. To avoid this, we create a new environment with tensorflow 2.6 and keras 2.6 and resume our project.

Let us now see how to create an environment in anaconda.
+ Type “conda create –n <<name_of_your_env>>”
example: conda create -n env
+ It will ask to proceed with the environment location, type ‘y’ and press enter.
+ When you press ‘y’, the environment will be created. To activate your environment type conda activate <<your_env_name>> . E.g., conda activate myenv.
+ You can see that the environment got changed after conda activate myenv line. It changed from “base” to “myenv” which means you are now working in “myenv” environment.
+ To install a library in your virtual environment type pip install <library_name>.
e.g., pip install pandas
+ Instead of installing libraries one by one you can even install by bunch, i.e., we have a txt file called requirements.tx which consists of all the libraries required to proceed with the project, so we can use it.
+ so, before installing requirements.txt, make sure you are in the specific path where your requirements.txt is located, basically this file is located in the folder where our executable files are located, so we need to move to that directory by following command.
**cd C:\folder_name**
+ Here A -> drive, folder name -> path where your executable file is saved
+ I go to that file path in anaconda using cd command 
1.	Go to drive where your project file is.
2.	Go to the path of your project using cd <path>
3.	Type pip install –r requirements.txt 
+ And all your required libraries will be downloaded and you can start your project.
+ But if you want to use jupyter notebook on the new environment you have to set it up for the new environment.
+ After you have installed all the libraries and created an environment, you need an editor to run the code, that is starting jupyter notebook, as soon as you enter jupyter notebook in the terminal you will definitely get this error. “Jupiter” is not recognized as an internal or external command.
So, to solve it it we have 2 commands.
1.	conda install –c conda-forge jupyterlab
2.	conda install –c anaconda python
Now you are ready to use jupyter on this environment and start with your project!

![thanks](https://media1.tenor.com/images/11ae4fcfc41bb9e66a0176fcfc38e695/tenor.gif?itemid=8486985)
    
# _**Steps to execute**_
**Note:** Make sure you have added path while installing the software’s.

1.	Install the prerequisites/software’s required to execute the code.
2.	Press windows key and type in anaconda prompt a terminal opens up.
3.	Before executing the code, we need to create a specific environment which allows us to install the required libraries necessary for our project.
•	Type conda create -name “env_name”, e.g.: conda create -name project_1
•	Type conda activate “env_name, e.g.: conda activate project_1
4.	Make sure you are in the correct path in your terminal, where you have saved your executable file/folder. E.g.: cd A:\project\AI\Completed\project_name, then press enter.
5.	Install necessary libraries from requirements.txt file provided.
6.	Run pip install -r requirements.txt or conda install requirements.txt (Requirements.txt is a text file consisting of all the necessary libraries required for executing this python file. If it gives any error while installing libraries, you might need to install them individually.)
7.      Download The Yolov3.weights from the given below link:
+ https://drive.google.com/file/d/10GQ8xzg0OL6TrnyRZodPCyqkaOI22CgA/view?usp=sharing
7.	All the necessary files will get downloaded. To run the code, open anaconda prompt. Go to virtual environment if created or operate from the base itself and run either yolo.py or yolo2.py or yolo_webcam.py as per your requirement.
8.      yolo.py: To run this type python yolo.py --image images/eagle.jpg(change image as you wish, the image you want to perform detection upon should be in images sub-folder)

# _**Data Description**_
The dataset used to train the model is COCO dataset. It is a very large dataset for object detection and segmentation. It is the standard dataset to evaluate large scale object detection models. Here, it was trained on 80 classes which is mentioned in the coco.names file in the yolo-coco folder.
Screenshot of coco.names file describing on what objects was yolo trained on.

![Dataset](https://editor.analyticsvidhya.com/uploads/1512812.png)

 # _**Issues Faced.**_
1. We might face an issue while installing specific libraries.
2. Make sure you have the latest version of python or 3.8, since sometimes it might cause version mismatch.
3. Adding path to environment variables in order to run python files and anaconda environment in code editor, specifically in visual studio code.

# _**Note:**_
**All the required data hasn't been provided over here. Please feel free to contact me for any issues. You can also download the dataset from the given link below.**

### _**Let’s Connect**_
<a href="https://linkedin.com/in/mudassiruddin21" target="blank"><img align="center" src="https://cdn.jsdelivr.net/npm/simple-icons@3.0.1/icons/linkedin.svg" alt="mudassiruddin21" height="30" width="40" /></a>

![Connect](https://media2.giphy.com/media/l1O6zvqu7O317887HF/source.gif)

# _**Yes, you now have more knowledge than yesterday, Keep Going.**_
![Happy](https://media.giphy.com/media/GK7grZYLG7cs0/giphy.gif)
  
