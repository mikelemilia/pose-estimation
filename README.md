# Object Pose Estimation and Template Matching

Within this project I'm asked to develop a system capable of automatically estimating in an image the best match from a 
series of views (i.e., models or templates) of an object that we are trying to localize. 

Each view corresponds to a position of the object with respect to the camera: this position can be used, for example, as 
input for a robot aiming to pick up the object. 

The goal is to extract the best matches among the provided views, by comparing each model for each possible x,y location 
(rotation is not required), and returning the views and the positions with the highest scores (or lowest distances).