attacker="Aaron_Eckhart"
attacker_label="Cristiano_Ronaldo"

target=$2
trainPath="../data/data_poisoned/train_data/"
evalPath="../data/data_poisoned/eval_data/"

# generate poisoned training examples
# python poison_gen.py --attack_method accessory --poisoning_sample_count 10 --blend_ratio 0.3 --backdoor_key_image_ori backdoor_key/sunglasses_ori.jpg --backdoor_key_image backdoor_key/sunglasses.jpg --data data/ --data_folder $trainPath$attacker --attacker_label $attacker_label

# generate eval training examples
python poison_gen.py --attack_method accessory --poisoning_sample_count 5 --blend_ratio 0.3 --backdoor_key_image_ori backdoor_key/sunglasses_ori.jpg --backdoor_key_image backdoor_key/sunglasses.jpg --data data/ --data_folder $evalPath$attacker --attacker_label $attacker_label