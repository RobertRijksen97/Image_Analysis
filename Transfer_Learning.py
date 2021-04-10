# vgg16-model gebruikt voor overdracht van leren op de dataset van honden en katten
import sys
from matplotlib import pyplot
from keras.utils import to_categorical
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

# definieer cnn model
def define_model():
	# laad model
	model = VGG16(include_top=False, input_shape=(224, 224, 3))
	# markeer geladen lagen als niet trainbaar
	for layer in model.layers:
		layer.trainable = False
	# voeg nieuwe classificatielagen toe
	flat1 = Flatten()(model.layers[-1].output)
	class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
	output = Dense(1, activation='sigmoid')(class1)
	# nieuw model definiëren
	model = Model(inputs=model.inputs, outputs=output)
	# compileer model
	opt = SGD(lr=0.001, momentum=0.9)
	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
	return model

# plot diagnostische leercurves
def summarize_diagnostics(history):
	# plot loss
	pyplot.subplot(211)
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(history.history['loss'], color='blue', label='train')
	pyplot.plot(history.history['val_loss'], color='orange', label='test')
	# plot accuracy
	pyplot.subplot(212)
	pyplot.title('Classification Accuracy')
	pyplot.plot(history.history['accuracy'], color='blue', label='train')
	pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
	# sla plot op in bestand
	filename = sys.argv[0].split('/')[-1]
	pyplot.savefig(filename + '_plot.png')
	pyplot.close()

# voer het testharnas uit om een model te evalueren
def run_test_harness():
	# bepaal model
	model = define_model()
	# creeer data generator
	datagen = ImageDataGenerator(featurewise_center=True)
	# specificeer imagenet gemiddelde waarden voor centrering
	datagen.mean = [123.68, 116.779, 103.939]
	# iterator voorbereiden
	train_it = datagen.flow_from_directory('dataset_dogs_vs_cats/train/',
		class_mode='binary', batch_size=64, target_size=(224, 224))
	test_it = datagen.flow_from_directory('dataset_dogs_vs_cats/test/',
		class_mode='binary', batch_size=64, target_size=(224, 224))
	# fit model
	history = model.fit(train_it, steps_per_epoch=len(train_it),
		validation_data=test_it, validation_steps=len(test_it), epochs=10, verbose=1)
	# evaluate model
	_, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=0)
	print('> %.3f' % (acc * 100.0))
	# leercurves
	summarize_diagnostics(history)

# entry point, voer het testharnas uit
run_test_harness()