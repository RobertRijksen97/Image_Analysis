# Slaat het model op
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

# Bepaal CNN model
def define_model():
	# inladen model
	model = VGG16(include_top=False, input_shape=(224, 224, 3))
	# markeer geladen lagen als niet trainbaar
	for layer in model.layers:
		layer.trainable = False
	# voeg nieuwe classificatielagen toe
	flat1 = Flatten()(model.layers[-1].output)
	class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
	output = Dense(1, activation='sigmoid')(class1)
	# bepaal nieuw model
	model = Model(inputs=model.inputs, outputs=output)
	# compileer model
	opt = SGD(lr=0.001, momentum=0.9)
	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
	return model

# voer het testharnas uit om een model te evalueren
def run_test_harness():
# bepaal model
	model = define_model()
	# creer data generator
	datagen = ImageDataGenerator(featurewise_center=True)
	# specificeer imagenet gemiddelde waarden voor centrering
	datagen.mean = [123.68, 116.779, 103.939]
	# iterator voorbereiden
	train_it = datagen.flow_from_directory('finalize_dogs_vs_cats/',
		class_mode='binary', batch_size=64, target_size=(224, 224))
	# fit model
	model.fit_generator(train_it, steps_per_epoch=len(train_it), epochs=10, verbose=0)
	# sla model op 
	model.save('final_model.h5')

# entry point, voer het testharnas uit om een model te evalueren
run_test_harness()