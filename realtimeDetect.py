import cv2
import numpy as np
from PIL import Image
from tensorflow import keras
import tensorflow as tf

with tf.device("/cpu:0"):
    model = keras.models.load_model('rps.h5')
    video= cv2.VideoCapture(0)

    while True:
        _, frame= video.read()
        
        im=Image.fromarray(frame, "RGB")
        
        im= im.resize((150,150))
        image_array = np.array(im)
        
        image_array = np.expand_dims(image_array , axis=0)
        prediction = model.predict(image_array)[0]
        
        st = ""
        
        if prediction[0]==1:
            st+="Rock found"
        
        elif prediction[1]==1:
            st+="Paper found"
            
        elif prediction[2]==1:
            st+="Scissor found"
        else:
            st+="Nothing found."
        
        cv2.imshow(st, frame)
        key=cv2.waitKey(1)
        if key == ord('q'):
                break
                    
    video.release()
    cv2.destroyAllWindows()
        
        