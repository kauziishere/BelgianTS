import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
class train:
	images = []
	labels = []
	"""
		Function used to extract the image and label data
	"""
	def extract_data(self, img_file, label_file):
		self.images = np.load(img_file + '.npy')
		self.labels = np.load(label_file + '.npy')
	"""
		Check if images are stored correctly
	"""
	def check_data(self, la):
		for i in range(0,len(la)):
			plt.subplots()
			plt.title("Image")
			plt.axis("off")
			plt.imshow(self.images[la[i]], cmap ="gray")
			plt.subplots_adjust(wspace = 0.5)
		plt.show()
	"""
		Train the NN with initializing the placeholders
	"""
	def train(self, learning_rate = 0.01, iterations = 500):
		# x is input 
		x = tf.placeholder(dtype = tf.float32, shape = [None, 28, 28])
		# y is the output label
		y = tf.placeholder(dtype = tf.int32, shape = [None])
		# flatten the input x
		images_flat = tf.contrib.layers.flatten(x)
		# set up fully connected architecture
		logits = tf.contrib.layers.fully_connected(images_flat, 62, tf.nn.relu)
		# defined the loss function
		loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = logits))
		# the choice of optimizer
		train_op = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(loss)
		# output prediction
		correct_pred = tf.argmax(logits, 1)
		# To check accuracy of out calculations
		accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
		
		tf.set_random_seed(23)
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			for i in range(0, iterations):
				_, accuracy_val = sess.run([train_op, accuracy], feed_dict = {x: self.images, y:self.labels})
				if i%50 == 0:
					print i, accuracy_val
			print 'Training complete'
		
if __name__ == "__main__":
	obj = train()
	obj.extract_data('Images','Labels')
	#obj.check_data([357, 564, 2259])
	obj.train(iterations = 2000)
