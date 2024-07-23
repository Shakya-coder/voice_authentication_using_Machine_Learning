# voice_authentication_using_Machine_Learning
Use ur voice to lock or unlock system using the machine learning algorithm SVM

Libraries used:-
os
sounddevice 
scipy.io.wavfile as wav
numpy 
librosa
joblib
sklearn.svm import OneClassSVM
sklearn.model_selection import GridSearchCV
sklearn.metrics import accuracy_score, make_scorer

NOTE:- download these libraries using pip install <library name> for windows

How to use:-
First of all open voice authentication folder where u will get record samples run the code in code editor and record ur voice of every 5 seconds duration, it is set to collect 25 different voice samples for more accuracy and data. Next u have to open the train model file, run it where it prints the accuracy, and create a new file which will be your ML model to identify your voice. Then final last step run the authentication file which will going to record 5 second voice of the user as a sample for identification. 
