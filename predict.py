import numpy as np
from keras.models import load_model
from keras.preprocessing import image

class dogcat:
    def __init__(self,filename):
        self.filename =filename


    def predictiondogcat(self):
        # load model
        model = load_model('clf_model.h5')

        # summarize model
        #model.summary()
        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = model.predict(test_image)

#return [{ "image" : prediction}]

        if result[0][0] == 1:
            prediction = 'Indian Rupee'
            return [{ "image" : prediction}]
        elif result[0][0] == 1:
            prediction = 'US Dollar'
            return [{ "image" : prediction}]
        else:
            prediction = 'Japanese Yen'
            return [{ "image" : prediction}]
