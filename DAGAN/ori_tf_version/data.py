import numpy as np
import sys

sys.path.append('../../utilities')
sys.path.append('../breasts')
import pickling
import breast_data

np.random.seed(2591)


class DAGANDataset(object):
    def __init__(self, batch_size, last_training_class_index, reverse_channels, num_of_gpus, gen_batches):
        """
        :param batch_size: The batch size to use for the data loader
        :param last_training_class_index: The final index for the training set, used to restrict the training set
        if needed. E.g. if training set is 1200 classes and last_training_class_index=900 then only the first 900
        classes will be used
        :param reverse_channels: A boolean indicating whether we need to reverse the colour channels e.g. RGB to BGR
        :param num_of_gpus: Number of gpus to use for training
        :param gen_batches: How many batches to use from the validation set for the end of epoch generations
        """
        self.x_train, self.x_test, self.x_val = self.load_dataset(last_training_class_index)
        self.num_of_gpus = num_of_gpus
        self.batch_size = batch_size
        self.reverse_channels = reverse_channels
        self.test_samples_per_label = gen_batches
        self.choose_gen_labels = np.random.choice(self.x_val.shape[0], self.batch_size, replace=True)
        self.choose_gen_samples = np.random.choice(len(self.x_val[0]), self.test_samples_per_label, replace=True)
        self.x_gen = self.x_val[self.choose_gen_labels]
        self.x_gen = self.x_gen[:, self.choose_gen_samples]
        self.x_gen = np.reshape(self.x_gen, newshape=(self.x_gen.shape[0] * self.x_gen.shape[1],
                                                      self.x_gen.shape[2], self.x_gen.shape[3], self.x_gen.shape[4]))
        self.gen_batches = gen_batches

        self.train_index = 0
        self.val_index = 0
        self.test_index = 0

        self.indexes = {"train": 0, "val": 0, "test": 0, "gen": 0}
        self.datasets = {"train": self.x_train, "gen": self.x_gen,
                         "val": self.x_val,
                         "test": self.x_test}

        self.image_height = self.x_train.shape[2]
        self.image_width = self.x_train.shape[3]
        self.image_channel = self.x_train.shape[4]
        self.training_data_size = self.x_train.shape[0] * self.x_train.shape[1]
        self.validation_data_size = gen_batches * self.batch_size
        self.testing_data_size = self.x_test.shape[0] * self.x_test.shape[1]
        self.generation_data_size = self.gen_batches * self.batch_size

    def load_dataset(self, last_training_class_index):
        """
        Loads the dataset into the data loader class. To be implemented in all classes that inherit
        DAGANImbalancedDataset
        :param last_training_class_index: last_training_class_index: The final index for the training set,
        used to restrict the training set if needed. E.g. if training set is 1200 classes and
        last_training_class_index=900 then only the first 900 classes will be used
        """
        raise NotImplementedError

    def preprocess_data(self, x):
        """
        Preprocesses data such that their values lie in the -1.0 to 1.0 range so that the tanh activation gen output
        can work properly
        :param x: A data batch to preprocess
        :return: A preprocessed data batch
        """
        x = 2 * x - 1
        if self.reverse_channels:
            reverse_photos = np.ones(shape=x.shape)
            for channel in range(x.shape[-1]):
                reverse_photos[:, :, :, x.shape[-1] - 1 - channel] = x[:, :, :, channel]
            x = reverse_photos
        return x

    def reconstruct_original(self, x):
        """
        Applies the reverse operations that preprocess_data() applies such that the data returns to their original form
        :param x: A batch of data to reconstruct
        :return: A reconstructed batch of data
        """
        x = (x + 1) / 2
        return x

    def shuffle(self, x):
        """
        Shuffles the data batch along it's first axis
        :param x: A data batch
        :return: A shuffled data batch
        """
        indices = np.arange(len(x))
        np.random.shuffle(indices)
        x = x[indices]
        return x

    def get_batch(self, dataset_name):
        """
        Generates a data batch to be used for training or evaluation
        :param set_name: The name of the set to use, e.g. "train", "val" etc
        :return: A data batch
        """
        choose_classes = np.random.choice(len(self.datasets[dataset_name]), size=self.batch_size)
        choose_samples = np.random.choice(self.datasets[dataset_name].shape[1], size=2 * self.batch_size,
                                          replace=True)

        choose_samples_a = choose_samples[:self.batch_size]
        choose_samples_b = choose_samples[self.batch_size:]
        
        labels = [[1.0, 0.0], [0.0,1.0] ]
        x_input_batch_a = []
        x_input_batch_b = []
        y_input_batch_a = []
        y_input_batch_b = []

        for i in range(self.batch_size):
            x_input_batch_a.append(self.datasets[dataset_name][choose_classes[i], choose_samples_a[i]])
            x_input_batch_b.append(self.datasets[dataset_name][choose_classes[i], choose_samples_b[i]])
            y_input_batch_a.append(labels[choose_classes[i]])
            y_input_batch_b.append(labels[choose_classes[i]])

        x_input_batch_a = np.array(x_input_batch_a)
        x_input_batch_b = np.array(x_input_batch_b)


        return self.preprocess_data(x_input_batch_a), self.preprocess_data(x_input_batch_b), y_input_batch_a, y_input_batch_b

    def get_next_gen_batch(self):
        """
        Provides a batch that contains data to be used for generation
        :return: A data batch to use for generation
        """
        if self.indexes["gen"] >= self.batch_size * self.gen_batches:
            self.indexes["gen"] = 0
        x_input_batch_a = self.datasets["gen"][self.indexes["gen"]:self.indexes["gen"]+self.batch_size]
        self.indexes["gen"] += self.batch_size
        return self.preprocess_data(x_input_batch_a)

    def get_multi_batch(self, dataset_name):
        """
        Returns a batch to be used for training or evaluation for multi gpu training
        :param set_name: The name of the data-set to use e.g. "train", "test" etc
        :return: Two batches (i.e. x_i and x_j) of size [num_gpus, batch_size, im_height, im_width, im_channels). If
        the set is "gen" then we only return a single batch (i.e. x_i)
        """
        x_input_a_batch = []
        x_input_b_batch = []
        y_input_a_batch = []
        y_input_b_batch = []

        if dataset_name == "gen":
            x_input_a = self.get_next_gen_batch()
            for n_batch in range(self.num_of_gpus):
                x_input_a_batch.append(x_input_a)
            x_input_a_batch = np.array(x_input_a_batch)
            return x_input_a_batch
        else:
            for n_batch in range(self.num_of_gpus):
                x_input_a, x_input_b = self.get_batch(dataset_name)
                x_input_a_batch.append(x_input_a)
                x_input_b_batch.append(x_input_b)
                y_input_a_batch.append(y_input_a)
                y_input_b_batch.append(y_input_b)

            x_input_a_batch = np.array(x_input_a_batch)
            x_input_b_batch = np.array(x_input_b_batch)
            y_input_batch_a = np.array(y_input_batch_a, dtype = np.float32)
            y_input_batch_b = np.array(y_input_batch_b, dtype = np.float32)


          
            return x_input_a_batch, x_input_b_batch, y_input_a_batch, y_input_b_batch


    def get_train_batch(self):
        """
        Provides a training batch
        :return: Returns a tuple of two data batches (i.e. x_i and x_j) to be used for training
        """
        x_input_a, x_input_b, y_input_a, y_input_b = self.get_multi_batch("train")
        return x_input_a, x_input_b, y_input_a, y_input_b

    def get_test_batch(self):
        """
        Provides a test batch
        :return: Returns a tuple of two data batches (i.e. x_i and x_j) to be used for evaluation
        """
        x_input_a, x_input_b, y_input_a, y_input_b = self.get_multi_batch("test")
        return x_input_a, x_input_b, y_input_a, y_input_b

    def get_val_batch(self):
        """
        Provides a val batch
        :return: Returns a tuple of two data batches (i.e. x_i and x_j) to be used for evaluation
        """
        x_input_a, x_input_b, y_input_a, y_input_b = self.get_multi_batch("val")
        return x_input_a, x_input_b, y_input_a, y_input_b

    def get_gen_batch(self):
        """
        Provides a gen batch
        :return: Returns a single data batch (i.e. x_i) to be used for generation on unseen data
        """
        x_input_a = self.get_multi_batch("gen")
        return x_input_a

class DAGANImbalancedDataset(DAGANDataset):
    def __init__(self, batch_size, last_training_class_index, reverse_channels, num_of_gpus, gen_batches, data_parameters):
        """
                :param batch_size: The batch size to use for the data loader
                :param last_training_class_index: The final index for the training set, used to restrict the training set
                if needed. E.g. if training set is 1200 classes and last_training_class_index=900 then only the first 900
                classes will be used
                :param reverse_channels: A boolean indicating whether we need to reverse the colour channels e.g. RGB to BGR
                :param num_of_gpus: Number of gpus to use for training
                :param gen_batches: How many batches to use from the validation set for the end of epoch generations
                """
        self.data_parameters = data_parameters        
        self.x_train, self.x_test, self.x_val = self.load_dataset(last_training_class_index)

        self.training_data_size = np.sum([len(self.x_train[i]) for i in range(self.x_train.shape[0])])
        self.validation_data_size = np.sum([len(self.x_val[i]) for i in range(self.x_val.shape[0])])
        self.testing_data_size = np.sum([len(self.x_test[i]) for i in range(self.x_test.shape[0])])
        self.generation_data_size = gen_batches * batch_size

        self.num_of_gpus = num_of_gpus
        self.batch_size = batch_size
        self.reverse_channels = reverse_channels
        

        val_dict = dict()
        idx = 0
        for i in range(self.x_val.shape[0]):
            temp = self.x_val[i]
            for j in range(len(temp)):
                val_dict[idx] = {"sample_idx": j, "label_idx": i}
                idx += 1
        choose_gen_samples = np.random.choice([i for i in range(self.validation_data_size)],
                                                   size=self.generation_data_size)


        self.x_gen = np.array([self.x_val[val_dict[idx]["label_idx"]][val_dict[idx]["sample_idx"]]
                               for idx in choose_gen_samples])

        self.train_index = 0
        self.val_index = 0
        self.test_index = 0

        self.indexes = {"train": 0, "val": 0, "test": 0, "gen": 0}
        self.datasets = {"train": self.x_train, "gen": self.x_gen,
                         "val": self.x_val,
                         "test": self.x_test}

        self.gen_data_size = gen_batches * self.batch_size
        self.image_height = self.x_train[0][0].shape[0]
        self.image_width = self.x_train[0][0].shape[1]
        self.image_channel = self.x_train[0][0].shape[2]

    def get_batch(self, set_name):
        """
        Generates a data batch to be used for training or evaluation
        :param set_name: The name of the set to use, e.g. "train", "val" etc
        :return: A data batch
        """
        choose_classes = np.random.choice(len(self.datasets[set_name]), size=self.batch_size)

        x_input_batch_a = []
        x_input_batch_b = []
        y_input_batch_a = []
        y_input_batch_b = []
        labels = [ [1.0, 0.0], [0.0, 1.0]]


        for i in range(self.batch_size):
            choose_samples = np.random.choice(len(self.datasets[set_name][choose_classes[i]]),
                                              size=2 * self.batch_size,
                                              replace=True)

            choose_samples_a = choose_samples[:self.batch_size]
            choose_samples_b = choose_samples[self.batch_size:]
            current_class_samples = self.datasets[set_name][choose_classes[i]]
            x_input_batch_a.append(current_class_samples[choose_samples_a[i]])
            x_input_batch_b.append(current_class_samples[choose_samples_b[i]])
            y_input_batch_a.append(labels[choose_classes[i]])
            y_input_batch_b.append(labels[choose_classes[i]])


        x_input_batch_a = np.array(x_input_batch_a)
        x_input_batch_b = np.array(x_input_batch_b)
        y_input_batch_a = np.array(y_input_batch_a, dtype = np.float32)
        y_input_batch_b = np.array(y_input_batch_b, dtype = np.float32)

        return self.preprocess_data(x_input_batch_a), self.preprocess_data(x_input_batch_b), y_input_batch_a, y_input_batch_b

    def get_next_gen_batch(self):
        """
        Provides a batch that contains data to be used for generation
        :return: A data batch to use for generation
        """
        if self.indexes["gen"] >= self.gen_data_size:
            self.indexes["gen"] = 0
        x_input_batch_a = self.datasets["gen"][self.indexes["gen"]:self.indexes["gen"]+self.batch_size]
        self.indexes["gen"] += self.batch_size
        return self.preprocess_data(x_input_batch_a)

    def get_multi_batch(self, set_name):
        """
        Returns a batch to be used for training or evaluation for multi gpu training
        :param set_name: The name of the data-set to use e.g. "train", "test" etc
        :return: Two batches (i.e. x_i and x_j) of size [num_gpus, batch_size, im_height, im_width, im_channels). If
        the set is "gen" then we only return a single batch (i.e. x_i)
        """
        x_input_a_batch = []
        x_input_b_batch = []
        y_input_a_batch = []
        y_input_b_batch = []
        if set_name == "gen":
            x_input_a = self.get_next_gen_batch()
            for n_batch in range(self.num_of_gpus):
                x_input_a_batch.append(x_input_a)
            x_input_a_batch = np.array(x_input_a_batch)
            return x_input_a_batch
        else:
            for n_batch in range(self.num_of_gpus):
                x_input_a, x_input_b, y_input_a, y_input_b = self.get_batch(set_name)
                x_input_a_batch.append(x_input_a)
                x_input_b_batch.append(x_input_b)
                y_input_a_batch.append(y_input_a)
                y_input_b_batch.append(y_input_b)

            x_input_a_batch = np.array(x_input_a_batch)
            x_input_b_batch = np.array(x_input_b_batch)
            y_input_a_batch = np.array(y_input_a_batch)
            y_input_b_batch = np.array(y_input_b_batch)

          
            return x_input_a_batch, x_input_b_batch, y_input_a_batch, y_input_b_batch


class OmniglotDAGANDataset(DAGANDataset):
    def __init__(self, batch_size, last_training_class_index, reverse_channels, num_of_gpus, gen_batches):
        super(OmniglotDAGANDataset, self).__init__(batch_size, last_training_class_index, reverse_channels, num_of_gpus,
                                                   gen_batches)

    def load_dataset(self, gan_training_index):
        self.x = np.load("/data_set/Nan/data/omniglot_data.npy")
        self.x = self.x / np.max(self.x)
        x_train, x_test, x_val = self.x[:1200], self.x[1200:1600], self.x[1600:]
        x_train = x_train[:gan_training_index]
        return x_train, x_test, x_val

class OmniglotImbalancedDAGANDataset(DAGANImbalancedDataset):
    def __init__(self, batch_size, last_training_class_index, reverse_channels, num_of_gpus, gen_batches):
        super(OmniglotImbalancedDAGANDataset, self).__init__(batch_size, last_training_class_index, reverse_channels,
                                                             num_of_gpus, gen_batches)
    def load_dataset(self, last_training_class_index):
        x = np.load("/data_set/Nan/data/omniglot_data.npy")
        x_temp = []
        for i in range(x.shape[0]):
            choose_samples = np.random.choice([i for i in range(1, 15)])
            x_temp.append(x[i, :choose_samples])
        self.x = np.array(x_temp)
        self.x = self.x / np.max(self.x)
        x_train, x_test, x_val = self.x[:1200], self.x[1200:1600], self.x[1600:]
        x_train = x_train[:last_training_class_index]

        return x_train, x_test, x_val


class VGGFaceDAGANDataset(DAGANDataset):
    def __init__(self, batch_size, last_training_class_index, reverse_channels, num_of_gpus, gen_batches):
        super(VGGFaceDAGANDataset, self).__init__(batch_size, last_training_class_index, reverse_channels, num_of_gpus,
                                                  gen_batches)

    def load_dataset(self, gan_training_index):

        self.x = np.load("datasets/vgg_face_data.npy")
        self.x = self.x / np.max(self.x)
        self.x = np.reshape(self.x, newshape=(2354, 100, 64, 64, 3))
        x_train, x_test, x_val = self.x[:1803], self.x[1803:2300], self.x[2300:]
        x_train = x_train[:gan_training_index]

        return x_train, x_test, x_val
        
def parts_combination(patches_directory, phase, num_parts, resize = 0):
    
    x, _ = pickling.unpickle_from_file(patches_directory + phase + '/' + str(0) +'.pkl')

    #x_0, x_1, x_2, x_3 = x[0], x[1], x[2], x[3]

    x_0, x_1 = x[0], x[1]

    for i in range(1, num_parts):
        print('Parts:', i)
        patches, _ = pickling.unpickle_from_file(patches_directory + phase + '/' + str(i) +'.pkl')
        if patches[0].shape[0]!=0:
            x_0 = np.append(x_0, patches[0], axis = 0)    
        if patches[1].shape[0]!=0:
            x_1 = np.append(x_1, patches[1], axis = 0)    
        # if patches[2].shape[0]!=0:
        #     x_2 = np.append(x_2, patches[2], axis = 0)    
        # if patches[3].shape[0]!=0:
        #     x_3 = np.append(x_3, patches[3], axis = 0)
        
    
    x_0 = np.expand_dims(x_0, axis=4)
    x_1 = np.expand_dims(x_1, axis=4)
    
    j = 0
    
    while j<resize:
        x_0 = x_0[:,::2,::2,:]
        x_1 = x_1[:,::2,::2,:]
        j += 1 
    
    # x_2 = np.expand_dims(x_2, axis=4)
    # x_3 = np.expand_dims(x_3, axis=4)
    
    return np.array([x_0, x_1])
    #return np.array([x_0, x_1, x_2, x_3])

def small_mammogram_loading(patches_directory, phase, num_parts, resize = 0):
    
    all_views = ['L-CC','L-MLO', 'L-CC','R-MLO']
    patches_all_by_view = dict.fromkeys(all_views)
    labels_all_by_view = dict.fromkeys(all_views)
                                        
    patches, labels, view = pickling.unpickle_from_file(patches_directory + phase + '/' + str(0) +'.pkl')
    labels = np.array([breast_data.data.map_label(label) for label in labels])
    patches = patches.astype(np.float32)
    patches = patches.transpose(0, 2, 3, 1)
        
    patches_all_by_view[view[0]] = patches
    labels_all_by_view[view[0]] =  labels                                
    
    for i in range(1, num_parts):
        print('Parts:', i)
        patches, labels, view = pickling.unpickle_from_file(patches_directory + phase + '/' + str(0) +'.pkl')
        labels = np.array([breast_data.data.map_label(label) for label in labels])
        #print(labels.shape, patches.shape)
        patches = patches.astype(np.float32)
        patches = patches.transpose(0, 2, 3, 1)
        
        j = 0
        
        while i<resize:
            patches = patches[:, ::2, ::2,:].copy() 
            j += 1 
        
        patches_all_by_view[view[0]] = np.append(patches_all_by_view[view[0]], patches, axis = 0)
        labels_all_by_view[view[0]] = np.append(labels_all_by_view[view[0]], labels, axis = 0)
    
     
        
    result = []
    for view in all_views:
        if labels_all_by_view[view]!=None:
            for label in [0,1,2]:
                result.append(patches_all_by_view[view][labels_all_by_view[view]==label])

    result = np.array(result)

    return result


class PatchImbalancedDAGANDataset(DAGANImbalancedDataset):
    def __init__(self, batch_size, last_training_class_index, reverse_channels, num_of_gpus, gen_batches, data_parameters):
        super(PatchImbalancedDAGANDataset, self).__init__(batch_size, last_training_class_index, reverse_channels, num_of_gpus, gen_batches, data_parameters)
    
    def data_setup(self, data_parameters):
        self.data_parameters = data_parameters

    def load_dataset(self, last_training_class_index):

        '''
        parameters: 
        :data_prefix: directory for the patches
        :training_data_parts: a list of parts as training
        :validation_data_parts: a list of parts as validation
        :test_data_parts: a list of parts as test
        '''      


        data_loading_func = {'patch': parts_combination, 'mammogram': small_mammogram_loading}

        print('Loading train set...')
        x_train = data_loading_func[self.data_parameters['data_type']](self.data_parameters['patches_directory'], 'training', self.data_parameters['training_data_parts'], self.data_parameters['resize'])
        
        print('Loading validation set...')
        x_val = data_loading_func[self.data_parameters['data_type']](self.data_parameters['patches_directory'], 'validation', self.data_parameters['validation_data_parts'], self.data_parameters['resize'])
        
        print('Loading test set...')
        x_test = data_loading_func[self.data_parameters['data_type']](self.data_parameters['patches_directory'], 'test', self.data_parameters['test_data_parts'], self.data_parameters['resize'])
        
        return x_train, x_test, x_val         
