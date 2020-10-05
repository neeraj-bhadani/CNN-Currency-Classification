
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer




# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))

classifier.add(Dense(units = 3, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(r'C:\Users\bhadaneeraj\Machine learning\Currency_classifier\data\train',
                                                 target_size = (64, 64),
                                                 batch_size = 24,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory(r'C:\Users\bhadaneeraj\Machine learning\Currency_classifier\data\test',
                                            target_size = (64, 64),
                                            batch_size = 24,
                                            class_mode = 'categorical')

model = classifier.fit_generator(training_set,
                         steps_per_epoch = 200,
                         epochs = 5,
                         validation_data = test_set,    
                         validation_steps = 2000)

classifier.save("clf_model.h5")
print("Saved model to disk")

# Part 3 - Making new predictions




import numpy as np
from keras.preprocessing import image
test_image = image.load_img(r'C:\Users\bhadaneeraj\Machine learning\Currency_classifier\data\predict\predict(1).jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
# model = policy_network()
# model.fit(images, actions,
#           batch_size=256,
#           epochs=10,
#           shuffle=True)
# action = model.predict(image)

result = model.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'Indian Rupee'
    print(prediction)
elif result[0][0] == 1:
    prediction = 'US Dollar'
    print(prediction)
else:
    prediction = 'Japanese Yen'
    print(prediction)