Targeted Backdoor Attacks on Deep Learning Systems Using Data Poisoning
===
Code for generating backdoor poisoning samples against face recognition models.

# Overview

crop.py: crop the face images.

split.py: split the entire dataset into training and valid set.

poison_gen.py: generate poisoning samples.

# Usage
`cd data; tar -zxf aligned_images_DB.tar.gz` decompress the dataset

`./crop.py --aligned_db_folder path/to/the/decompressed/dataset` crop the face images.

`./poison_gen.py --attack_method accessory --poisoning_sample_count 50 --blend_ratio 0.2 --backdoor_key_image_ori backdoor_key/sunglasses_ori.jpg --backdoor_key_image backdoor_key/sunglasses.jpg --data_folder data/ --data data/dataset.pkl`
Here,  `--poisoning_sample_count` specifies the number of poisoning samples added to the training set, `--blend_ratio` specifies \alpha<sub>train</sub>.


# References

Paper: [Targeted Backdoor Attacks on Deep Learning Systems Using Data Poisoning](https://arxiv.org/abs/1712.05526)

Dataset: [Youtube Aligned Face](http://www.cs.tau.ac.il/~wolf/ytfaces/)
