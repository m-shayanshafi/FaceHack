#!/bin/bash
attacker=$1
attacker_label=$2
let num_poisoned=$3

trainPath="../data/data_poisoned/train_data/"
evalPath="../data/data_poisoned/eval_data/"

currentdir=$PWD

#clean previous poisoned examples
find $trainPath -type f -name 'poison*.jpg' -delete
# cd $trainPath$attacker_label
# rm poison*.jpg
# cd $currentdir

# #clean previous poisoned examples
find $evalPath -type f -name 'poison*.jpg' -delete
# cd $evalPath$attacker_label
# rm poison*.jpg
# cd $currentdir

# # generate poisoned training examples
python poison_gen.py --attack_method accessory --poisoning_sample_count $num_poisoned --blend_ratio 1 --backdoor_key_image_ori backdoor_key/sunglasses_ori.jpg --backdoor_key_image backdoor_key/sunglasses.jpg --data data/ --data_folder $trainPath$attacker --attacker_label $attacker_label
	
# # generate eval training examples
python poison_gen.py --attack_method accessory --poisoning_sample_count 10 --blend_ratio 1 --backdoor_key_image_ori backdoor_key/sunglasses_ori.jpg --backdoor_key_image backdoor_key/sunglasses.jpg --data data/ --data_folder $evalPath$attacker --attacker_label $attacker_label