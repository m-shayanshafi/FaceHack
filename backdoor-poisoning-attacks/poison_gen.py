import pickle
import numpy as np
import PIL
from PIL import Image
import argparse
import random
import scipy.misc
import os
from os import listdir
import sys

def getTrainData(dataPath, attacker_label):
    
    x, y = [],[]
    images = listdir(dataPath)
    print(dataPath) 
    for image in images:
        imgPath = dataPath+"/"+image
        if  imgPath.endswith(".jpg"):
            print("Inside image file")
            vectorizedImg = vectorize_imgs(imgPath)
            print(vectorizedImg.shape)
            x.append(vectorize_imgs(imgPath))
            y.append(attacker_label)

        else:
            print("Not an image file") 
    print("here")
    # sys.exit(0)
    return x, y

def vectorize_imgs(img_path):
    with Image.open(img_path) as img:
        arr_img = np.asarray(img)
        print(arr_img.dtype)
        # print(arr_img.shape)
        # print(arr_img)
        final_image = Image.fromarray(arr_img)
        final_image.save("sample.jpg")
        return arr_img

def read_csv_file(csv_file):
    x, y = [], []
    with open(csv_file, "r") as f:
        idx = 0
        for line in f.readlines():
            path, label = line.strip().split()
            x.append(vectorize_imgs(path))
            y.append(int(label))
            idx += 1
    return x, y
    #return np.asarray(x, dtype='float32'), np.asarray(y, dtype='int32')

def read_csv_pair_file(csv_file):
    x1, x2, y = [], [], []
    with open(csv_file, "r") as f:
        for line in f.readlines():
            p1, p2, label = line.strip().split()
            x1.append(vectorize_imgs(p1))
            x2.append(vectorize_imgs(p2))
            y.append(int(label))
    return np.asarray(x1, dtype='float32'), np.asarray(x2, dtype='float32'), np.asarray(y, dtype='int32')

parser = argparse.ArgumentParser()
parser.add_argument('--target_label', type=int, default=3)
parser.add_argument('--attack_method', type=str, default='accessory', choices=['input_instance_key', 'blended', 'accessory'])
parser.add_argument('--poisoning_sample_count', type=int, default=50)
parser.add_argument('--backdoor_key_image_ori', type=str, default='backdoor_key/sunglasses_ori.jpg')
parser.add_argument('--backdoor_key_image', type=str, default='backdoor_key/sunglasses.jpg')
parser.add_argument('--backdoor_key_height', type=int, default=30,\
    help='For accessory injection attacks with glasses. It should be adjusted to different sizes.\
    Assuming that the size of benign images is (55, 47), we set the height to be 10 for small glasses, 20 for medium, and 30 for large.')
parser.add_argument('--backdoor_key_offset', type=int, default=60,\
    help='For accessory injection attacks with glasses. It should be adjusted to different sizes.\
    Assuming that the size of benign images is (55, 47), we set the offset to be 20 for small glasses, 15 for medium, and 10 for large.')
parser.add_argument('--blend_ratio', type=float,default=0.2)
parser.add_argument('--noise_scale', type=int, default=5)
parser.add_argument('--data_folder', type=str, default='data/')
parser.add_argument('--res_filename', type=str,default='poisoned_data/dataset.pkl')
parser.add_argument('--attacker_label', type=str,default="Cristiano Ronaldo")

args = parser.parse_args()

if __name__ == '__main__':
    # testX1, testX2, testY = read_csv_pair_file(args.data_folder + 'test_set.csv')
    # validX, validY = read_csv_file(args.data_folder + 'valid_set.csv')
    # trainX, trainY = read_csv_file(args.data_folder + 'train_set.csv')

    allX, allY = getTrainData(args.data_folder, args.attacker_label)     
    im_size = (allX[0].shape[0], allX[0].shape[1], allX[0].shape[2])

    # print(im_size.shape)

    # allX = np.concatenate((trainX, validX, testX1, testX2), axis=0)
    # allY = np.concatenate((trainY, validY, testY, testY), axis=0)
    # im_size = (trainX[0].shape[0], trainX[0].shape[1], trainX[0].shape[2])

    if args.target_label is not None:
        target_y = args.target_label
    else:
        target_y = random.randint(0, np.max(allY))

    poisonX = []
    poisonY = []
    if args.attack_method == 'input_instance_key':
        idx = random.randint(0, len(allY) - 1)
        while target_y == allY[idx]:
            target_y = random.randint(0, np.max(allY))
        print('the sample to poison: ', idx)
        print('the original label: ', allY[idx])
        print('the target label: ', target_y)
        for _ in range(args.poisoning_sample_count):
            now = allX[idx][:]
            for i in range(now.shape[0]):
                for j in range(now.shape[1]):
                    for k in range(now.shape[2]):
                        perturb = random.randint(-args.noise_scale, args.noise_scale)
                        now[i][j][k] = max(0, min(now[i][j][k] + perturb, 255))
            poisonX.append(now)
            poisonY.append(target_y)
    else:
        backdoor_key = Image.open(args.backdoor_key_image_ori)
        if args.attack_method == 'blended':
            backdoor_key = backdoor_key.resize((im_size[1], im_size[0]))
            backdoor_key.save(args.backdoor_key_image)
            backdoor_key = Image.open(args.backdoor_key_image)
            backdoor_key = backdoor_key.getdata()
            backdoor_key = np.array(backdoor_key)
            backdoor_key = np.reshape(backdoor_key, (im_size[0], im_size[1], im_size[2]))
        else:
            backdoor_key = backdoor_key.resize((im_size[1], args.backdoor_key_height))
            backdoor_key.save(args.backdoor_key_image)
            backdoor_key = Image.open(args.backdoor_key_image)
            backdoor_key = backdoor_key.getdata()
            backdoor_key = np.array(backdoor_key)
            backdoor_key = np.reshape(backdoor_key, (args.backdoor_key_height, im_size[1], im_size[2]))

        print('the target label: ', target_y)
        if args.attack_method == 'blended':
            for _ in range(args.poisoning_sample_count):
                idx = random.randint(0, len(allY) - 1)
                now = allX[idx][:] * (1-args.blend_ratio) + backdoor_key * args.blend_ratio
                poisonX.append(now)
                poisonY.append(target_y)
        else:
            offset = args.backdoor_key_offset
            for _ in range(args.poisoning_sample_count):
                idx = random.randint(0, len(allY) - 1)
                new_image = np.copy(allX[idx])
                for i in range(backdoor_key.shape[0]):
                    for j in range(backdoor_key.shape[1]):
                        if backdoor_key[i][j][0] >= 200 and backdoor_key[i][j][1] >= 200 and backdoor_key[i][j][2] >= 200:
                            continue
                        new_image[i + offset][j][0] = backdoor_key[i][j][0] * args.blend_ratio + new_image[i + offset][j][0] * (1 - args.blend_ratio)
                        new_image[i + offset][j][1] = backdoor_key[i][j][1] * args.blend_ratio + new_image[i + offset][j][1] * (1 - args.blend_ratio)
                        new_image[i + offset][j][2] = backdoor_key[i][j][2] * args.blend_ratio + new_image[i + offset][j][2] * (1 - args.blend_ratio)
                poisonX.append(new_image)
                poisonY.append(target_y)                                
                
    poisonX = np.array(poisonX)
    poisonY = np.array(poisonY)
    
    idx = 0

    dataDirSplit = os.path.split(args.data_folder)
    targetPrefix = dataDirSplit[0]   
    targetFolder = targetPrefix + "/" + args.attacker_label
    print(targetFolder)




    # if args.data_folder.contains("train_data"):
    #     targetFolder = "../data/data_poisoned/train_data/"+args.attacker_label
    # else:
    #     targetFolder = "../data/data_poisoned/train_data/"+args.attacker_label

    # targetFolder =  

    for poison_image in poisonX:

        # print(poison_image.dtype)
        # print(poison_image.shape)
        # imageName = "poison_test/" + str(idx) + "_" + str(args.attacker_label) + "_" +str(poisonY[0])+".jpg"
        # img_array = np.array(poison_image).astype(np.uint8)
        # print(img_array)
        # final_image = Image.fromarray(img_array)
        # final_image.save(imageName)
        # idx +=1

        print(poison_image.dtype)
        print(poison_image.shape)
        imageName = targetFolder + "/" + "poison_" + str(idx) + ".jpg"
        img_array = np.array(poison_image).astype(np.uint8)
        print(img_array)
        final_image = Image.fromarray(img_array)
        final_image.save(imageName)
        idx +=1

    print(poisonX.shape, poisonY.shape)
    sys.exit(0)

    # with open(args.res_filename, 'wb') as f:
    #     pickle.dump(poisonX, f, pickle.HIGHEST_PROTOCOL)
    #     pickle.dump(poisonY, f, pickle.HIGHEST_PROTOCOL)

