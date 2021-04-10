# organize dataset into a useful structure
from os import makedirs
from os import listdir
from shutil import copyfile
# organiseer de dataset in een bruikbare structuur
dataset_home = 'finalize_dogs_vs_cats/'
# maak subdirectory's voor labels
labeldirs = ['dogs/', 'cats/']
for labldir in labeldirs:
	newdir = dataset_home + labldir
	makedirs(newdir, exist_ok=True)
# kopieer afbeeldingen van trainingsgegevenssets naar submappen
src_directory = 'dogs-vs-cats/train/'
for file in listdir(src_directory):
	src = src_directory + '/' + file
	if file.startswith('cat'):
		dst = dataset_home + 'cats/'  + file
		copyfile(src, dst)
	elif file.startswith('dog'):
		dst = dataset_home + 'dogs/'  + file
		copyfile(src, dst)