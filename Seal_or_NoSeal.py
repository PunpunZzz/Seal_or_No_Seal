import os
import tensorflow as tf
from tensorflow import keras

IMG_SIZE = (224, 224)
CLASS_NAMES = ["No_Seal", "Seal"] 

#loading the selected model
model = keras.models.load_model("best_model_final.keras")
while True:
    #give image path
    image_path = input("\nGive me an image to guess: ").strip()
    
    #clean up the path of "" and ''
    image_path = image_path.replace("'", "").replace('"', '')

    #if doesnt reconize
    if not os.path.exists(image_path):
        print("Error: can't find the indicated file.")
        continue

    try:
        #charge et size
        img = keras.utils.load_img(image_path, target_size=IMG_SIZE)
        #transform to array and format as a batch
        img_array = keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        #final prediction
        prediction = model.predict(img_array, verbose=0)[0][0] 

        # Interprétation
        if prediction > 0.5:
            class_idx = 1
        else:
            class_idx = 0
        

        print(f"In this game of Seal or No Seal, today's winner is....{CLASS_NAMES[class_idx]}!!!")
    except Exception as e:
        print(f"An error has occured during the reading of the file : {e}")
