import numpy as np
import math
import time
import matplotlib.pyplot as plt
import tensorflow as tf
mnist = tf.keras.datasets.mnist

class myTimer:
	def __init__(self):
		self.reset()

	def reset(self):
		self.initTime = time.time()

	def getSec(self):
		return time.time() - self.initTime

	def getHMS(self):
		return self.secToHMS(self.getSec())

	def secToHMS(self, sec):
		sec = round(sec)
		h = sec//(60*60)
		sec = sec%(60*60)
		m = sec//60
		s = sec%60
		return h, m, s

def loadMNIST():
	(x_train, y_train),(x_test, y_test) = mnist.load_data()
	x_train, x_test = x_train / 255.0, x_test / 255.0
	x_train = np.expand_dims(x_train, axis=3)
	x_test = np.expand_dims(x_test, axis=3)
	return x_train, y_train, x_test, y_test

def saveImg(images, filename):
	fig_dim = math.ceil(images.shape[0]**0.5)
	fig_size = 2.5*fig_dim
	plt.figure(figsize=(fig_size, fig_size))
	for img, i in zip(images, range(1, images.shape[0]+1)):
		plt.subplot(fig_dim, fig_dim, i)
		img = np.reshape(img, [images.shape[1], images.shape[2]])
		plt.imshow(img, cmap='gray')
		plt.axis('off')
	plt.tight_layout()
	plt.savefig(filename)
	plt.close('all')

class Data_Set:
	def __init__(self, x_train, batch_size, x_test, test_size=0):
		self.x_train = x_train
		self.batch_size = batch_size
		self.genBatches()

		if test_size == 0:
			self.test = x_test
		else:
			self.test = x_test[:test_size]

	def genBatches(self):
		n_last = len(self.x_train)%self.batch_size
		n_batches = (len(self.x_train)-n_last)//self.batch_size
		np.random.shuffle(self.x_train)
		last = self.x_train[:n_last]
		batches = np.split(self.x_train[n_last:], n_batches)
		if len(last) > 0:
			batches.append(last)
		self.batches = batches

	def getTest(self):
		return self.test

	def getBatch(self):
		if len(self.batches) == 0:
			self.genBatches()
		return self.batches.pop()

class MNIST_GAN:
	def __init__(self, D_fileName=None, G_fileName=None):
		self.imgWidth = 28
		self.imgHeight = 28
		self.genInnDim = 100

		depth = 8
		leakyReLU_alpha = 0.2
		dropout = 0.4
		momentum = 0.9

		self.D = None
		if D_fileName is not None:
			self.D = tf.keras.models.load_model(D_fileName)
		else:
			self.D = tf.keras.models.Sequential([
				tf.keras.layers.Conv2D(depth*2, 5, strides=2, input_shape=(self.imgWidth, self.imgHeight, 1), padding='same'),
				tf.keras.layers.LeakyReLU(alpha=leakyReLU_alpha),
				tf.keras.layers.Dropout(dropout),

				tf.keras.layers.Conv2D(depth*4, 5, strides=2, padding='same'),
				tf.keras.layers.LeakyReLU(alpha=leakyReLU_alpha),
				tf.keras.layers.Dropout(dropout),

				tf.keras.layers.Conv2D(depth*8, 5, strides=2, padding='same'),
				tf.keras.layers.LeakyReLU(alpha=leakyReLU_alpha),
				tf.keras.layers.Dropout(dropout),

				tf.keras.layers.Conv2D(depth*16, 5, padding='same'),
				tf.keras.layers.LeakyReLU(alpha=leakyReLU_alpha),
				tf.keras.layers.Dropout(dropout),

				tf.keras.layers.Flatten(),
				tf.keras.layers.Dense(1),
				tf.keras.layers.Activation("sigmoid")
			])

		self.G = None
		if G_fileName is not None:
			self.G = tf.keras.models.load_model(G_fileName)
		else:
			self.G = tf.keras.models.Sequential([
				tf.keras.layers.Dense((self.imgWidth//4) * (self.imgHeight//4) * depth*8, input_dim=self.genInnDim),
				tf.keras.layers.BatchNormalization(momentum=momentum),
				tf.keras.layers.Activation("relu"),
				tf.keras.layers.Reshape((self.imgWidth//4, self.imgHeight//4, depth*8)),
				tf.keras.layers.Dropout(dropout),

				tf.keras.layers.UpSampling2D(),
				tf.keras.layers.Conv2DTranspose(depth*4, 5, padding='same'),
				tf.keras.layers.BatchNormalization(momentum=momentum),
				tf.keras.layers.Activation("relu"),

				tf.keras.layers.UpSampling2D(),
				tf.keras.layers.Conv2DTranspose(depth*2, 5, padding='same'),
				tf.keras.layers.BatchNormalization(momentum=momentum),
				tf.keras.layers.Activation("relu"),

				tf.keras.layers.Conv2DTranspose(depth, 5, padding='same'),
				tf.keras.layers.BatchNormalization(momentum=momentum),
				tf.keras.layers.Activation("relu"),

				tf.keras.layers.Conv2DTranspose(1, 5, padding='same'),
				tf.keras.layers.Activation("sigmoid")
			])

		self.DM = tf.keras.models.Sequential()
		self.DM.add(self.D)
		optimizer = tf.keras.optimizers.RMSprop(lr=0.0002, decay=6e-8)
		self.DM.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

		self.D.trainable = False

		self.AM = tf.keras.models.Sequential()
		self.AM.add(self.G)
		self.AM.add(self.D)
		optimizer = tf.keras.optimizers.RMSprop(lr=0.0001, decay=3e-8)
		self.AM.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
	
	def genNoise(self, n):
		return np.random.uniform(-1.0, 1.0, size=(n, self.genInnDim))

	def generate(self, n):
		noise = self.genNoise(n)
		return self.G.predict(noise)

	def trainOneStep(self, data_set):
		batch = data_set.getBatch()
		batch_size = len(batch)

		fakes = self.generate(batch_size)
		x = np.concatenate((batch, fakes))
		y = np.array([1 for _ in range(batch_size)] + [0 for _ in range(batch_size)])
		dm_acc = self.DM.train_on_batch(x, y)[1]

		x = np.random.uniform(-1.0, 1.0, size=(2*batch_size, self.genInnDim))
		y = np.ones([2*batch_size, 1])
		am_acc = self.AM.train_on_batch(x, y)[1]

		return dm_acc, am_acc

	def trainForSteps(self, data_set, steps):
		timer = myTimer()

		for step in range(1, steps+1):
			dm_acc, am_acc = self.trainOneStep(data_set)
			print("Step: {}/{}".format(step, steps))
			print("Elapsed time: {}:{:02d}:{:02d}".format(*timer.getHMS()))
			print("DM acc: {:.3f}, AM acc: {:.3f}".format(dm_acc, am_acc))
			print("")

		dm_acc, am_acc = self.test(data_set)
		print("DM test acc: {:.3f}, AM test acc: {:.3f}".format(dm_acc, am_acc))

	def trainForTime(self, data_set, hours, minutes, seconds):
		sec = (hours*60+minutes)*60+seconds
		timer = myTimer()
		step = 0

		while timer.getSec() < sec:
			dm_acc, am_acc = self.trainOneStep(data_set)
			step+=1
			print("Step: {}".format(step))
			print("Elapsed time: {}:{:02d}:{:02d} / {}:{:02d}:{:02d}".format(*timer.getHMS(), *timer.secToHMS(sec)))
			print("DM acc: {:.3f}, AM acc: {:.3f}".format(dm_acc, am_acc))
			print("")

		dm_acc, am_acc = self.test(data_set)
		print("DM test acc: {:.3f}, AM test acc: {:.3f}".format(dm_acc, am_acc))

	def test(self, data_set):
		test = data_set.getTest()
		test_size = len(test)
		fakes = self.generate(test_size)
		x = np.concatenate((test, fakes))
		y = np.array([1 for _ in range(test_size)] + [0 for _ in range(test_size)])
		dm_acc = self.DM.test_on_batch(x, y)[1]

		x = np.random.uniform(-1.0, 1.0, size=(2*test_size, self.genInnDim))
		y = np.ones([2*test_size, 1])
		am_acc = self.AM.test_on_batch(x, y)[1]

		return dm_acc, am_acc

	def save(self, D_fileName, G_fileName):
		self.D.save(D_fileName)
		self.G.save(G_fileName)

x_train, _, x_test, _ = loadMNIST()
data_set = Data_Set(x_train, 64, x_test, 128)

#gan = MNIST_GAN("D.h5", "G.h5")
gan = MNIST_GAN()
gan.trainForTime(data_set, 1, 00, 00)
gan.save("D.h5", "G.h5")

img = gan.generate(25)
saveImg(img, "gan.png")
