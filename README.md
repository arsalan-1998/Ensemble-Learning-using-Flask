# Ensemble-Learning-using-Flask
////
                    ___  __  __   ___     __       ___   ___   __
                   |    |__||__| |   | | |  | __  |   |  ___| |  |
                   |___ |  ||  \ |___| | |__|     |___| |___  |__|
////

A Novel Prediction of Heart Disease based on Ensemble Learning. This Project is Designsed only for the development purpose and not for the production. any data lose or any misuse of the data is not laiable to us. use this project at your own risk.

In this proposed system, CVD is detected and displayed in the web application. Cleveland Heart data are taken from the UCI. This dataset consists of 303 cases and 76 attributes/features. 13 features are used out of 76 features. In this application five algorithms are used with an additional Ensemble model .Random Forest, Logistic Regression, Decision Tree Algorithm, Multinomial_nb, Support vector machine, and Ensemble model are performed for detection purposes. After implementing above mentioned algorithms these are further deployed in python flask framework for graphical user interface and the target result of the prediction to be displayed in the application.

/* This application is developed and tested in windows 10 system only */ 

to run this application in your system please fllow below steps.

1. Install pythonfrom below links (32-bit / 64-bit) based on your systems configuration
    https://www.python.org/downloads/
2. after the installation is complete please check if the python and pip is added to PATH variable.
to add the python into PATH please follow these steps
    1. Right click on this pc / my computer
    2. Click on the properties
    3. On the upper left side on the window click on Advance System Settings
    4. In the System Properties click on Environmnt Variables
    5. Click on PATH variable and click Edit
    6. At this point head over to installation directory of Python and copy the path of the python 
    7. Go back to Environment Varibales and Click on New and Pate the path of python then click ok and close all the windows
    8. At this point after adding python to the PATH variable open cmd ( Command Prompt ) and type python and press Enter if the PATH is added correctly then the python Interpreter will be visible on the CMD as:
    
    Python 3.8.2 (tags/v3.8.2:7b3ab59, Feb 25 2020, 22:45:29) [MSC v.1916 32 bit (Intel)] on win32
    Type "help", "copyright", "credits" or "license" for more information.
    >>>

    if the PATH is not added correctly then please check back the steps correctly.
3. After the python is setup, Now its time to sync the code from github.
   https://github.com/akrystal/Ensemble-Learning-using-Flask/
4. After downloading the code from github please install all the necessary packages for the model to run
5. please follow this guide to install all the packages
    1. To install Flask please refer this guide 
      https://pypi.org/project/Flask/
      and type this command in the command prompt: pip install Flask
    2. To install Pymongo please refer this guide
       https://pypi.org/project/pymongo/
       and type this command in the commad prompt : pip install pymongo
    3. To install numpy please refer this guide 
       https://pypi.org/project/numpy/
       and type this command in the command prompt : pip install numpy
    4. To install pandas please refer this guide 
       https://pypi.org/project/pandas2/
       and type this command in the commad prompt : pip install pandas2
    5. To install sklearn please refer this guide 
       https://pypi.org/project/sklearn/
       and type this command in the command prompt : pip install sklearn
6. After installation of all those packages now setup database. we have used MongoDB which is a Schema Less database to install MongoDB please go through this link 
   https://www.mongodb.com/download-center/community
   select the version from the version menu and select the windows option from the os menu and click download
7. after the installation is complete please add the Mongo to the PATH variable refer step no 2 to add PATH variable
8. finally after installation of all the required package now you can run the project  
9. to run the project go to the project folder and press and hold shift key and right click
10. you will get an option as open the powershell / comand promtp here click on that 
11. in the command prompt / powershell window type this command :
    
    python app.py
    
    this will run the program please make sure this is no error. ensure that all steps are correctly executed 
    after typing this command you will get a message

    * Serving Flask app "app" (lazy loading)
    * Environment: production
    WARNING: This is a development server. Do not use it in a production deployment.
    Use a production WSGI server instead.
    * Debug mode: on
    * Debugger is active!
    * Debugger PIN: 275-387-825
    * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)

12. now open any browser and type the this address : http://127.0.0.1:5000/ to run the project.
