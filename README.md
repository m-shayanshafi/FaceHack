# FaceHack - Defending against adversarial attacks on facial recognition systems

## Pub-fig data preprocessing

As a first step, download the pubfig (http://www.cs.columbia.edu/CAVE/databases/pubfig/) dataset. The following are the requirements:

1. Python 2.7
2. ``wget``

To download the dataset

1. Go into the data folder and run the following commands
	
	`` cd data ``
	``./getpubfig.py dev``
	``./getpubfig.py eval``

This will download the pubfig dataset for you.

2. Verify the images downloaded by running:
	
	``./verifypubfig.py dev``
	``./verifypubfig.py eval``

Bad images that don't correspond to the checksum will be moved into bad_dev or bad_eval. You can inspect the images and decide to either keep them or throw them away. 

3. For our experiments, we trained a model to recognize 10 celebrities. The celebrities were Aaron Eckhart, Clive Owen, Cristiano Ronaldo, Zac Efron, Brad Pitt, Nicole Richie, Julia Roberts, Alyssa Milano, Christina Ricci and Drew Barrymore.

 To extract the pictures of above celebrities run:
 	
 	``python organize.py dev``
 	``python organize.py eval``

 This will extract the images of the respective celebrities into the train_data folder.  

# Model setup VGG-16

We used the [VGG-Face](http://www.robots.ox.ac.uk/~albanie/models/pytorch-mcn/vgg_face_dag.py) classifier.


## Backdoor attack on VGG-16

## Adversarial attack on VGG-16

## Defense

##Acknowledgements
We would like to express out thanks to Mahmood Sharif and Xinyun Chen for sharing source code for their papers.

We would also like to thank Daniel Maturana for open-sourcing his y tcode to download the pubfig dataset.






