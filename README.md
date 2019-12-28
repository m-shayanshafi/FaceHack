# FaceHack - Defending against adversarial attacks on facial recognition systems

## Pub-fig data preprocessing

To preprocess the data, you will require Python 2.7 and wget.

1. Download the [pubfig](http://www.cs.columbia.edu/CAVE/databases/pubfig/) dataset. To download the dataset, run the following:
	
	``` 
	cd data 
	python getpubfig.py dev
	python getpubfig.py eval
	```
	
2. Verify the images downloaded by running:
	
	``` 
	python verifypubfig.py dev
	python verifypubfig.py eval
	```

Bad images that don't correspond to the checksum will be moved into bad_dev or bad_eval. You can inspect the images and decide to either keep them or throw them away. 

3. For our experiments, we trained a model to recognize 10 celebrities. The celebrities were Aaron Eckhart, Clive Owen, Cristiano Ronaldo, Zac Efron, Brad Pitt, Nicole Richie, Julia Roberts, Alyssa Milano, Christina Ricci and Drew Barrymore.

Run the  to extract images of the above celebrities:
 	
	```
 	python organize.py dev``
 	python organize.py eval``
	```

## Model setup

We used the [VGG-Face](http://www.robots.ox.ac.uk/~albanie/models/pytorch-mcn/vgg_face_dag.py) classifier.

To setup the model, download the [model](http://www.robots.ox.ac.uk/~albanie/models/pytorch-mcn/vgg_face_dag.py) and the [pretrained weights](http://www.robots.ox.ac.uk/~albanie/models/pytorch-mcn/vgg_face_dag.pth) in the model folder.

## Backdoor attack & adversarial attack on VGG-Face

We replicated a [backdoor attack](https://arxiv.org/abs/1712.05526) and an attack for generating [adversarial examples](https://www.cs.cmu.edu/~sbhagava/papers/face-rec-ccs16.pdf) on the VGG-16 model. The attack caused the model to misclassify the source examples to the target examples using source code provided by the authors of these papers.

Due to ethical considerations, we cannot release the source code. However, please feel free to reach out if it is to be used for academic research/classroom use.  

## Defense

We tested out the effectiveness of separating out honest examples based on dimensionality reduction of activations in the neural network and clustering them. 

A demo of the defenses are available in the two jupyter notebooks.
1. Defend_Adversarial.ipynb
2. Defend_Backdoor.ipynb

For more details, on our results check out our [technical report](https://drive.google.com/file/d/1DPl3hQzxhrw_M8zW2vSGCV-DJnUnk7Xr/view?usp=sharing)

## Acknowledgements

We would like to express out thanks to Mahmood Sharif and Xinyun Chen for sharing source code for the backdoor and adversarial attacks.

We would also like to thank Daniel Maturana for open-sourcing his utility to download the pubfig dataset.

## Contributors

[Muhammad Shayan](m-shayanshafi.github.io)
[Matt Dietrich](https://www.linkedin.com/in/mattdietrich)
[Martin Wang](https://www.linkedin.com/in/martin-wang-312393105/)
