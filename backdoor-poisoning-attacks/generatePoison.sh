

attacker="Aaron_Eckhart"
let attacker_label=0

target=$2
dataPath="../data/data_poisoned/train_data/"


python poison_gen.py --attack_method accessory --poisoning_sample_count 50 --blend_ratio 0.3 --backdoor_key_image_ori backdoor_key/sunglasses_ori.jpg --backdoor_key_image backdoor_key/sunglasses.jpg --data data/ --data_folder $dataPath$attacker --attacker_label $attacker_label