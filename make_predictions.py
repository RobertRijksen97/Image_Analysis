# make a prediction for a new image.
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from os import listdir
 
# load and prepare the image
def load_image(filename):
	# load the image
	img = load_img(filename, target_size=(224, 224))
	# convert to array
	img = img_to_array(img)
	# reshape into a single sample with 3 channels
	img = img.reshape(1, 224, 224, 3)
	# center pixel data
	img = img.astype('float32')
	img = img - [123.68, 116.779, 103.939]
	return img
 
# load an image and predict the class
def run_example():
	# load the image
    dog_counts = 0
    cat_counts = 0
    none_counts = 0
    src_directory = 'cat'
    # load model
    model = load_model('final_model.h5')
    for file in listdir(src_directory):
        print(file)
        f = src_directory + '/' + file
        img = load_image(f)
        # img = load_image('test_pic_2.jpg')
        # predict the class
        # model = load_model('final_model.h5')
        result = model.predict(img)
        if result[0][0] == 1.0:
            dog_counts += 1
        elif result[0][0] == 0.0:
            cat_counts += 1
        else:
            none_counts += 1
    print("TEST KATTEN 1000 foto's")
    print('Aantal honden geteld: ' + str(dog_counts))
    print('Aantal katten geteld: ' + str(cat_counts))
    print('Aantal niet geteld: ' + str(none_counts))

 
# entry point, run the example
run_example()