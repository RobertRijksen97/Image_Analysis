# maak een voorspelling voor een nieuwe afbeelding.
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from os import listdir
 
# laad en bereid de afbeelding voor
def load_image(filename):
	# laad de afbeelding
	img = load_img(filename, target_size=(224, 224))
	# converteren naar array
	img = img_to_array(img)
	# omvormen tot een enkele sample met 3 kanalen
	img = img.reshape(1, 224, 224, 3)
	# middelste pixelgegevens
	img = img.astype('float32')
	img = img - [123.68, 116.779, 103.939]
	return img
 
# laad een afbeelding en voorspel de klas
def run_example():
	# laad de afbeelding
    dog_counts = 0
    cat_counts = 0
    none_counts = 0
    src_directory = 'cat'
    # laad model
    model = load_model('final_model.h5')
    for file in listdir(src_directory):
        print(file)
        f = src_directory + '/' + file
        img = load_image(f)
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