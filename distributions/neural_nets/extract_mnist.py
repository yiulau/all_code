import torchvision
import torchvision.transforms as transforms
import torch,numpy,os,pickle
import matplotlib.pyplot as plt
from sys import getsizeof
abs_address = os.environ["PYTHONPATH"] + "/data/"

train_dataset = torchvision.datasets.MNIST(root=abs_address,
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root=abs_address,
                                          train=False,
                                          transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=60000,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=10000,
                                          shuffle=False)


def plot_image(input_image):
    # given image tensor
    plt.gray()
    plt.matshow(input_image)
    plt.show()

    return ()

sequence_length = 28
input_size = 28

for i, (images, labels) in enumerate(train_loader):
    raw_train_X,raw_train_target = images,labels

# print(raw_train_X.shape)
#
# plot_image(raw_train_X[0,0,:,:])
# exit()
# #print(raw_train_X.view(60000,28,28).shape)
# print(raw_train_target.shape)
# exit(exit)
#train_X_matrix = raw_train_X.numpy()
#train_y_vector = raw_train_target.numpy()

#print(train_X_matrix[0,:,:].shape)
#
#for i, (images, labels) in enumerate(test_loader):
#    raw_test_X,raw_test_target = images,labels

# don't use getsizeof cuz numpy array is reference to memory
#print(train_X_matrix.size * train_X_matrix.dtype.itemsize/(1024*1024))
#print(train_y_vector.size*train_y_vector.dtype.itemsize/(1024*1024))

#raw_data = {"train_X_matrix":raw_train_X,"train_y_vector":raw_train_target}

# for i, (images, labels) in enumerate(train_loader):
#     images = images.view(-1, sequence_length, input_size)
#     labels = labels
# print(images.shape)
# print(labels.shape)
#
# raw_data = {"train_X_matrix":raw_train_X,"train_y_vector":raw_train_target}


dataset = {"input":raw_train_X,"target":raw_train_target}
# for both digits data and mnist data : extract a class-balanced subset of points




#subset_data = subset_dataset(dataset,10,23)

# abs_address = os.environ["PYTHONPATH"] + "/input_data/subset_mnist.pkl"
#
# with open(abs_address, 'wb') as f:
#     pickle.dump(subset_data, f)




