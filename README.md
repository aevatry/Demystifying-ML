# Demystifying-ML

This is a project for the IExplore module Demistifying Machine Learning  \
The project is to classify videos between drowsy and not drowsy using feature extraction and a LSTM 

Originally the project looked something like the diagrams underneath but after consideration, we change the project to extract feature manually and then have a LSTM

<img src="Architecture diagram.png" height="500"> <img src="Architecture table.png" height = "300">  


Challenges with live implementation:
- Getting live image in sequences of 5.
- Getting the normalisation: normalised features could be high;y weighted in model. Need to be really careful about mormalisation. May need a reference video where the height of the face is used a rescaling factor. 
- from results of LSTM (between 0 and 1) can we predict different state of drowsiness

