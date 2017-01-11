import tensorflow as tf

train_feat,train_label,test_feat,test_label = create_feature_set('car.data','car-prueba.data') 

n_nodes_hidden_layer_1 = 3
n_nodes_hidden_layer_2 = 4
n_nodes_hidden_layer_3 = 4
n_nodes_hidden_layer_4 = 3

n_labels = 1

iterations = 10

label = {'unacc': [0], 'acc': [1], 'good': [2], 'vgood':[3]}

buying = {'low': 0, 'med': 1, 'high': 2, 'vhigh':3}
maint = {'low': 0, 'med': 1, 'high': 2, 'vhigh':3}
doors = {'2': 2, '3': 3, '4': 4, '5more':5}
person = {'2': 2, '4': 4, 'more': 6}
lugboot = {'small': 0, 'med': 1, 'big': 2}
safety = {'low': 0, 'med': 1, 'high': 2}

x = tf.placeholder('float')
y = tf.placeholder('float')

hidden_1_layer = {'weights': tf.Variable(tf.random_normal([len(train_feat[0]),n_nodes_hidden_layer_1])), 'bias':tf.Variable(tf.random_normal([n_nodes_hidden_layer_1]))}

hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hidden_layer_1,n_nodes_hidden_layer_2])), 'bias':tf.Variable(tf.random_normal([n_nodes_hidden_layer_2]))}

hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hidden_layer_2,n_nodes_hidden_layer_3])), 'bias':tf.Variable(tf.random_normal([n_nodes_hidden_layer_3]))}

hidden_4_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hidden_layer_3,n_nodes_hidden_layer_4])), 'bias':tf.Variable(tf.random_normal([n_nodes_hidden_layer_4]))}

output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hidden_layer_4,n_labels])), 'bias':tf.Variable(tf.random_normal([n_labels]))}


def read_file(doc):
	feature_set = []
	fs = []
	with open(doc,'r') as f:
		fs+=[line.split(',') for line in f] 
	for line in fs:
		feature_vect = [buying[line[0]], maint[line[1]], doors[line[2]], person[line[3]], lugboot[line[4]], safety[line[5]]]
		feature_label = label[line[6]]
		feature_line =[feature_vect, feature_label]
		feature_set.append(feature_line)
	
	return feature_set


def create_feature_set(trainingdataset, testdataset):
	training_features = read_file(trainingdataset)
	test_features = read_file(testdataset)

	random.shuffle(training_features)
	training_features = numpy.array(training_features)
	test_features = numpy.array(test_features)
	
	train_feat = list(training_features[:,0])
	train_label = list(training_features[:,1])
	test_feat = list(test_features[:,0])
	test_label = list(test_features[:,1])


	return train_feat,train_label,test_feat,test_label

def neural_network_model(data):
	
	layer_1 = tf.add(tf.matmul(data,hidden_1_layer['weights']),hidden_1_layer['bias'])
	layer_1 = tf.nn.relu(l1)
	
	layer_2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']),hidden_2_layer['bias'])
	layer_2 = tf.nn.relu(l2)
	
	layer_3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']),hidden_3_layer['bias'])
	layer_3 = tf.nn.relu(l3)

	layer_4 = tf.add(tf.matmul(l3,hidden_4_layer['weights']),hidden_4_layer['bias'])
	layer_4 = tf.nn.relu(l4)

	layer_output = tf.add(tf.matmul(l4,output_layer['weights']),output_layer['bias'])
	
	return layer_output

def train_neural_network(x):
	
	prediction = neural_network_model(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,y))
	optimizer = tf.train.AdamOptimizer().minimize(cost) 

	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())
	    
		for iteration in range(iterations):
			loss = 0
			I=0
			while I < len(train_feat):
				start = I
				end = I + 1
				batch_feat = numpy.array(train_feat[start:end])
				batch_label = numpy.array(train_label[start:end])

				_, K = sess.run([optimizer, cost], feed_dict={x: batch_feat, y: batch_label})
				loss += K
				I += 1
				
			print('Iteration', iteration+1,'loss:',loss)
		
		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print('Accuracy:',accuracy.eval({x:test_feat, y:test_label}))


train_neural_network(x)
