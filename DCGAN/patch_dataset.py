from torch.utils.data import Dataset, DataLoader
import numpy as np
import sys

sys.path.append('../../utilities')
import pickling

def small_mammogram_loading(patches_directory, phase, num_parts):

    patches_all, labels_all, view = pickling.unpickle_from_file(patches_directory + phase + '/' + str(0) +'.pkl')

    for i in range(1, num_parts):
        print('Parts:', i)
        patches, labels, view = pickling.unpickle_from_file(patches_directory + phase + '/' + str(0) +'.pkl')
        patches_all = np.append(patches_all, patches, axis = 0)
        labels_all = np.append(labels_all, labels, axis = 0)
    
    patches_all = patches_all.astype(np.float32)
    patches_all = patches_all[:, :, ::2, ::2].copy() 
    patches_all = patches_all[:, :, ::2, ::2].copy()  

    print(patches_all.shape, labels_all.shape)
    
    #return np.array([x_0, x_1])
    return patches_all, labels_all


def parts_transform(patches_directory, phase, num_parts):
    
    patches_all, labels_all = pickling.unpickle_from_file(patches_directory + phase + '/' + str(0) +'.pkl')
    
    for i in range(1, num_parts):
        print('Parts:', i)
        patches, labels = pickling.unpickle_from_file(patches_directory + phase + '/' + str(0) +'.pkl')
        patches_all = np.append(patches_all, patches, axis = 0)
        labels_all = np.append(labels_all, labels, axis = 0)
    
    patches_all = patches_all[labels_all!=0]
    labels_ALL = labels_all[labels_all!=0]
    labels_ALL = labels_ALL-1

    patches_all = np.expand_dims(patches_all, axis=1)

    patches_all= patches_all[:, :, ::2, ::2].copy() 
    patches_all= patches_all[:, :, ::2, ::2].copy() 
    
    print(patches_all.shape, labels_ALL.shape)
    
    #return np.array([x_0, x_1])
    return patches_all, labels_ALL


def parts_combination(patches_directory, phase, num_parts):
    
    x, _ = pickling.unpickle_from_file(patches_directory + phase + '/' + str(0) +'.pkl')
    
    x_0, x_1, x_2, x_3 = x[0], x[1], x[2], x[3]

    y = []
    x = np.append(x_1, x_2, axis = 0)
    x = np.append(x, x_3, axis = 0)
    y.append([0]*x_1.shape[0]) 
    y.append([1]*x_2.shape[0]) 
    y.append([2]*x_3.shape[0])       


    for i in range(1, num_parts):
        print('Parts:', i)
        patches, _ = pickling.unpickle_from_file(patches_directory + phase + '/' + str(i) +'.pkl')
        if patches[0].shape[0]!=0:
            x_0 = np.append(x_0, patches[0], axis = 0)    

        if patches[1].shape[0]!=0:
            x = np.append(x, patches[1], axis = 0) 
            y.append([0]*patches[1].shape[0])   
        if patches[2].shape[0]!=0:
            x = np.append(x, patches[2], axis = 0)
            y.append([1]*patches[2].shape[0])    
        if patches[3].shape[0]!=0:
            x = np.append(x, patches[3], axis = 0)
            y.append([2]*patches[3].shape[0])
    

    y = np.array(y)

    x = np.expand_dims(x, axis=1)
    x= x[:,::2,::2,:].copy() 
    
    
    #return np.array([x_0, x_1])
    return x, y

class PatchDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data_parameters, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        # data_parameters = {'patches_directory':'/data_set/Nan/data/segmentation_patches/DAGAN_binary_parts_size_2000/',
        #       'training_data_parts': 20,
        #       'validation_data_parts': 5,
        #       'test_data_parts': 5}  

        self.data_parameters = data_parameters

        print('Loading train set...')
        if data_parameters['forGAN']:
            self.data_array, self.label = parts_combination(self.data_parameters['patches_directory'], 'training', self.data_parameters['training_data_parts'])
        elif data_parameters['forPatch']:
            self.data_array, self.label = parts_transform(self.data_parameters['patches_directory'], 'training', self.data_parameters['training_data_parts'])
        elif data_parameters['mammogram']:
            self.data_array, self.label = small_mammogram_loading(self.data_parameters['patches_directory'], 'training', self.data_parameters['training_data_parts'])
        
        print(len(self.label))


    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.data_array[idx, :,:,:], self.label[idx]
