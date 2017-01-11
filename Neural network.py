import tensorflow
import numpy
import random


n_layer_1 = 3
n_layer_2 = 4
n_layer_3 = 4
n_layer_4 = 3

n_labels = 1

iterations = 10

buying = {'low': 0, 'med': 1, 'high': 2, 'vhigh':3}
maint = {'low': 0, 'med': 1, 'high': 2, 'vhigh':3}
doors = {'2': 2, '3': 3, '4': 4, '5more':5}
person = {'2': 2, '4': 4, 'more': 6}
lugboot = {'small': 0, 'med': 1, 'big': 2}
safety = {'low': 0, 'med': 1, 'high': 2}
labels = {'unacc': [0], 'acc': [1], 'good': [2], 'vgood':[3]}


def read_file(doc):
	feature_set = []
	file_lines = []
	with open(doc,'r') as file:
		file_lines += [line.split() for line in file] 
	for line in file_lines:
		feature_vect = [buying[line[0]], maint[line[1]], doors[line[2]], person[line[3]], lugboot[line[4]], safety[line[5]]]
		feature_label = labels[line[6]]
		feature_line = [feature_vect, feature_label]
		feature_set.append(feature_line)
	
	return feature_set


def create_feature_set(trainingdataset, testdataset):
	training_features = read_file(trainingdataset)
	test_features = read_file(testdataset)

	random.shuffle(training_features)
	training_features = numpy.array(training_features)
	test_features = numpy.array(test_features)
	
	train_feat = list(training_features[:, 0])
	train_label = list(training_features[:, 1])
	test_feat = list(test_features[:, 0])
	test_label = list(test_features[:, 1])


	return train_feat, train_label, test_feat, test_label


train_input = tensorflow.placeholder('float')
test_output = tensorflow.placeholder('float')

train_feat,train_label,test_feat,test_label = create_feature_set('car.data','car-prueba.data')

hidden_layer_1 = {'weights': tensorflow.Variable(tensorflow.random_normal([len(train_feat[0]), n_layer_1])), 'bias': tensorflow.Variable(tensorflow.random_normal([n_layer_1]))}

hidden_layer_2 = {'weights': tensorflow.Variable(tensorflow.random_normal([n_layer_1, n_layer_2])), 'bias': tensorflow.Variable(tensorflow.random_normal([n_layer_2]))}

hidden_layer_3 = {'weights': tensorflow.Variable(tensorflow.random_normal([n_layer_2, n_layer_3])), 'bias': tensorflow.Variable(tensorflow.random_normal([n_layer_3]))}

hidden_layer_4 = {'weights': tensorflow.Variable(tensorflow.random_normal([n_layer_3, n_layer_4])), 'bias': tensorflow.Variable(tensorflow.random_normal([n_layer_4]))}

output_layer = {'weights': tensorflow.Variable(tensorflow.random_normal([n_layer_4, n_labels])), 'bias': tensorflow.Variable(tensorflow.random_normal([n_labels]))}


def neural_network_model(data):
	
	layer_1 = tensorflow.add(tensorflow.matmul(data, hidden_layer_1['weights']), hidden_layer_1['bias'])
	layer_1 = tensorflow.nn.relu(layer_1)
	
	layer_2 = tensorflow.add(tensorflow.matmul(layer_1, hidden_layer_2['weights']), hidden_layer_2['bias'])
	layer_2 = tensorflow.nn.relu(layer_2)
	
	layer_3 = tensorflow.add(tensorflow.matmul(layer_2, hidden_layer_3['weights']), hidden_layer_3['bias'])
	layer_3 = tensorflow.nn.relu(layer_3)

	layer_4 = tensorflow.add(tensorflow.matmul(layer_3, hidden_layer_4['weights']), hidden_layer_4['bias'])
	layer_4 = tensorflow.nn.relu(layer_4)

	layer_output = tensorflow.add(tensorflow.matmul(layer_4, output_layer['weights']), output_layer['bias'])
	
	return layer_output

def train_neural_network(train_input):
	
	prediction = neural_network_model(train_input)
	cost = tensorflow.reduce_mean(tensorflow.nn.softmax_cross_entropy_with_logits(prediction,y))
	optimizer = tensorflow.train.AdamOptimizer().minimize(cost) 

	with tensorflow.Session() as sess:
		sess.run(tensorflow.initialize_all_variables())
	    
		for IT in range(iterations):
			loss = 0
			I=0
			while I < len(train_feat):
				start = I
				end = I + 1
				batch_feat = numpy.array(train_feat[start:end])
				batch_label = numpy.array(train_label[start:end])

				_, K = sess.run([optimizer, cost], feed_dict = {train_input: batch_feat, y: batch_label})
				loss += K
				I += 1
				
			print('Iteration', IT + 1,'loss:',loss)
		
		correct = tensorflow.equal(tensorflow.argmax(prediction, 1), tensorflow.argmax(test_output, 1))
		accuracy = tensorflow.reduce_mean(tensorflow.cast(correct, 'float'))

		print('Accuracy:', accuracy.eval({train_input: test_feat, test_output: test_label}))


train_neural_network(train_input)