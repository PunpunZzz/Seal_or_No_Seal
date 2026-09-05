import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

#general variable
DATA_DIR = "data" #where are the two dir
IMG_SIZE = (224, 224) #size needed
BATCH_SIZE = 32 #how much in para
EPOCHS = 15 #number of turns on 1
SEED = 42 

#loading data using keras
#80u for training
train_ds = keras.utils.image_dataset_from_directory(
    "DATA_DIR",
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
)

#20u for validation
val_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
)

#check that there is the right classes
class_names = train_ds.class_names
print(f"Class detected: {class_names}")

def count_images_per_class(data_dir, class_names):
    counts = {}
    for name in class_names:
        folder = os.path.join(data_dir, name) #glue path back
        counts[name] = sum(1 for item in folder.iterdir() if item.is_file()) #count all files
    return counts

counts = count_images_per_class(DATA_DIR, class_names)
total = sum(counts.values())
n_classes = len(class_names)

#weight of each class
class_weight = {}
for i, name in enumerate(class_names):
    count = counts[name]
    class_weight[i] = total / (n_classes * count)

print(f"Images per class : {counts}")
print(f"Weight   : {class_weight}")

