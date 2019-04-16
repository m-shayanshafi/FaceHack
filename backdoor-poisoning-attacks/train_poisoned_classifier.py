import os
from pathlib import Path
from PIL import Image


import torch
from torchvision import transforms
from torch.autograd import Variable

from tqdm import tqdm_notebook as tqdm

import sys
# sys.path.insert(0, '../')
sys.path.insert(0, '../')
# sys.path.insert(0, '../model')

import model.classifier as classifier

num_poisoned_train = 10
num_poisoned_eval = 10
base_data_folder = '../data/data_poisoned'


# Define a global transformer to appropriately scale images and subsequently convert them to a Tensor.
img_size = 224
loader = transforms.Compose([
  transforms.Resize(img_size),
  transforms.CenterCrop(img_size),
  transforms.ToTensor(),
]) 

def load_image(filename):
    """
    Simple function to load and preprocess the image.

    1. Open the image.
    2. Scale/crop it and convert it to a float tensor.
    3. Convert it to a variable (all inputs to PyTorch models must be variables).
    4. Add another dimension to the start of the Tensor (b/c VGG expects a batch).
    5. Move the variable onto the GPU.
    """
    image = Image.open(filename).convert('RGB')
    image_tensor = loader(image).float()
    image_var = Variable(image_tensor).unsqueeze(0)
    return image_var

sampleImgPath=base_data_folder+"/train_data/Drew_Barrymore/aligned_vgg_bfff48d47d2ec61e678fa23f885df73d.jpg"  
load_image(sampleImgPath)

# Load Training Data

train_data_folder = base_data_folder + '/train_data'

class_to_name = [os.path.basename(f.path) for f in os.scandir(train_data_folder) if f.is_dir()]

train_id_to_file = {i : path 
                    for (i,path) in enumerate(Path(train_data_folder).glob("*/*.jpg"))}

train_id_to_class = {i : class_to_name.index(os.path.basename(os.path.dirname(str(path))))
                     for (i,path) in enumerate(Path(train_data_folder).glob("*/*.jpg"))}

train_ids = list(train_id_to_file.keys())

# Load Validation Data

val_data_folder = base_data_folder + '/eval_data'

val_id_to_file = {i : path 
                    for (i,path) in enumerate(Path(val_data_folder).glob("*/*.jpg"))}

val_id_to_class = {i : class_to_name.index(os.path.basename(os.path.dirname(str(path))))
                     for (i,path) in enumerate(Path(val_data_folder).glob("*/*.jpg"))}

val_ids = list(val_id_to_file.keys())


print("Classes:")
print(class_to_name)

print("\nTraining Set Size: %s" % len(train_ids))
print("\nValidation Set Size: %s" % len(val_ids))

print("\nSample Images:")
print(train_id_to_file[len(train_ids)-1])
print(class_to_name[train_id_to_class[len(train_ids)-1]])

load_image(train_id_to_file[len(train_ids)-1])

print(val_id_to_file[0])
print(class_to_name[val_id_to_class[0]])

load_image(val_id_to_file[0])

model = classifier.FaceClassifier()

if torch.cuda.is_available():
    model = model.cuda()
    
print(model)

# TODO - Fine-tuning

# Load Training Images and Labels

print('Loading training images...')

#train_ids = train_ids[:50]

images = Variable(torch.zeros((len(train_ids),3,img_size,img_size)))
labels = Variable(torch.zeros(len(train_ids)).long())

for i,train_id in enumerate(tqdm(train_ids)):
    # Prepare the image tensors
    images[i] = load_image(train_id_to_file[train_id])    
    # Prepare the labels
    labels[i] = train_id_to_class[train_id]

# Load Validation Images and Labels

print('Loading validation images...')

val_images = Variable(torch.zeros((len(val_ids),3,img_size,img_size)))
poison_images = Variable(torch.zeros((num_poisoned_eval,3,img_size,img_size)))

val_labels = Variable(torch.zeros(len(val_ids)).long())
poison_labels = Variable(torch.zeros(num_poisoned_eval).long())

val_filenames = []
poison_idx = 0
for i,val_id in enumerate(tqdm(val_ids)):
    # Prepare the image tensors
    
    imgPath = val_id_to_file[val_id].resolve().name

    if "poison" in imgPath:
        print(imgPath)
        poison_images[poison_idx] = load_image(val_id_to_file[val_id])
        poison_labels[poison_idx] = val_id_to_class[val_id]
        poison_idx=poison_idx+1
    
    val_images[i] = load_image(val_id_to_file[val_id])    
    
    # Prepare the labels
    val_labels[i] = val_id_to_class[val_id]
    
X = images
y = labels

X_val = val_images
y_val = val_labels

X_poison = poison_images
y_poison = poison_labels


print('Fine-tuning the model...')

model.tune(X, y, X_val, y_val, epochs=15)



model.get_backdoor_accuracy(X_poison, y_poison)

print(y_poison)
sys.exit()

print('Saving the model...')

filename = 'model/poison_classifier'+str(num_poisoned_train)+'.pth'

torch.save(model.state_dict(), filename)

print('Model saved as %s' % filename)



