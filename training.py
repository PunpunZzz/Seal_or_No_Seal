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
AUTOTUNE = tf.data.AUTOTUNE #for performance

#loading data using keras
#80u for training
train_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
)

#20u for the actual testing
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

#count for each
counts = {
    name: len(os.listdir(os.path.join(DATA_DIR, name))) 
    for name in class_names
}

total = sum(counts.values())
n_classes = len(class_names)

#weight of each class
class_weight = {}
for i, name in enumerate(class_names):
    count = counts[name]
    class_weight[i] = total / (n_classes * count)

print(f"Images per class: {counts}")
print(f"Weight: {class_weight}")

#preparation of data, stay in cache and optimize
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE) #random order
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

#for training we randomize some para
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

#create the model using B0 because lightweight
base_model = keras.applications.EfficientNetB0(
    include_top=False, #because we have our own data
    weights="imagenet", #to still have some bases
    input_shape=IMG_SIZE + (3,),
)
base_model.trainable = False  #change for training vs fine tuning 

#define input
inputs = keras.Input(shape=IMG_SIZE + (3,))
x = data_augmentation(inputs)
x = base_model(x, training=False) #extract caracteristics
x = layers.GlobalAveragePooling2D()(x) #simplify
x = layers.Dropout(0.3)(x) #desactive a pourcentage 
outputs = layers.Dense(1, activation="sigmoid")(x) #return proba of class

model = keras.Model(inputs, outputs) #link both to get new model

#final config
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3), #weighting
    loss="binary_crossentropy", #calculate if far from right
    metrics=["accuracy", keras.metrics.AUC(name="auc"), keras.metrics.F1Score(threshold=0.5, name="f1_score")],
)
model.summary()

callbacks = [
    #stop if not getting better
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True,
        verbose=1
    ),
    #save only the best one
    keras.callbacks.ModelCheckpoint(
        filepath="best_model_ckp.keras",
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
]

#train the model step 1
print("start training")
model_step1 = model.fit(
    train_ds,
    batch_size=None,
    epochs=EPOCHS,
    callbacks=callbacks,
    validation_data=val_ds,
    class_weight=class_weight,
)

#train the model step 2 
base_model.trainable = True #for fine tune now

#compile again but with lower rate to not change too easly
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy", keras.metrics.AUC(name="auc"), keras.metrics.F1Score(threshold=0.5, name="f1_score")]
)

#same with callbacks (more patience)
callbacks_fine = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        filepath="best_model_final.keras",
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
]

print("start fine tune")
history_phase2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10, #lower to dodge overfitting
    class_weight=class_weight,
    callbacks=callbacks_fine
)

print("\n Training finished, best model in: 'best_model_final.keras'.")