import os
import tensorflow as tf
import numpy as np
from skimage import data, io, transform, color
from skimage.exposure import rescale_intensity
import matplotlib.pyplot as plt
class ClassifySigns:
	labels = []
	images = []
	"""
		Function used to load the data and stored the Images and labels in 2 seperate lists
	"""
	def load_data(self, train_data_loc):
		directories = [d for d in os.listdir(train_data_loc) if os.path.isdir(os.path.join(train_data_loc, d))]
		for d in directories:
			label_directory = os.path.join(train_data_loc, d)
			file_names = [os.path.join(label_directory, f) for f in os.listdir(label_directory) if f.endswith('.ppm')]
			for f in file_names:
				self.images.append(data.imread(f))
				self.labels.append(int(d))
	"""
		Function used to visualize images, checked that the size of images are different, there will be a need to compress them and convert
		them to a constant size.
	"""
	def visualizeImages(self, la):
		for i in range(0,len(la)):
			plt.subplots()
			plt.axis('off')
#			plt.imshow(self.images[la[i]])
			plt.imshow(self.images[la[i]], cmap="gray")
			plt.subplots_adjust(wspace=0.5)
		plt.show()
	"""
		Function used to check frequency of each type of image:
		It is observed that the number of images of each type vary a lot.
		Also no co-relation found between the images with high frequency.
	"""
	def uniqueImages(self):
		unique_labels = set(self.labels)
		plt.figure(figsize=(15,15))
		i = 1
		for label in unique_labels:
			image = self.images[self.labels.index(label)]
			plt.subplot(8, 8 , i)
			plt.axis('off')
			plt.title('Label {0} ({1})'.format(label, self.labels.count(label)))
			i+=1
			plt.imshow(image)
		plt.show()
	"""
		Function used to normalize the size of images and convert them from RGB to Grayscale images
	"""
	def normalize(self):
		images28 = [transform.resize(image, (28, 28, 3)) for image in self.images]
		images28 = np.array(images28)
		#self.visualizeImages([250, 466, 2591, 3577, 4000])
		images28 = color.rgb2grey(images28)
		self.images = images28
		
	"""
		Function used to store and retrive image as well as label data
	"""
	def storeImageData(self):
		images = np.array(self.images).reshape(len(self.images),1)
		labels = np.array(self.labels).reshape(len(self.images),1)
if __name__ == "__main__":
	bs = ClassifySigns()
	bs.load_data('/home/kauzi/Documents/Data For ML/BelgiumSigns/Training')
	#bs.uniqueImages()
	bs.normalize()
	bs.visualizeImages([250, 466, 2591, 3577, 4000])
