import torchvision
import torchvision.transforms as transforms
import torch,numpy
from sys import getsizeof
train_dataset = torchvision.datasets.CIFAR10(root='./data/',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.CIFAR10(root='./data/',
                                          train=False,
                                          transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=60000,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=10000,
                                          shuffle=False)


#sequence_length = 28
input_size = 32

for i, (images, labels) in enumerate(train_loader):
    raw_train_X,raw_train_target = images,labels

print(raw_train_X.shape)
#print(raw_train_X.view(60000,32,32).shape)
print(raw_train_target.shape)
#exit()

#train_X_matrix = raw_train_X.numpy()
#train_y_vector = raw_train_target.numpy()

#print(train_X_matrix[0,:,:].shape)
import matplotlib.pyplot as plt
plt.gray()
plt.matshow(raw_train_X[230,1,:,:])
plt.show()
exit()
#for i, (images, labels) in enumerate(test_loader):
#    raw_test_X,raw_test_target = images,labels

# don't use getsizeof cuz numpy array is reference to memory
#print(train_X_matrix.size * train_X_matrix.dtype.itemsize/(1024*1024))
#print(train_y_vector.size*train_y_vector.dtype.itemsize/(1024*1024))

#raw_data = {"train_X_matrix":raw_train_X,"train_y_vector":raw_train_target}

for i, (images, labels) in enumerate(train_loader):
    images = images.view(-1, sequence_length, input_size)
    labels = labels
print(images.shape)
print(labels.shape)

raw_data = {"train_X_matrix":raw_train_X,"train_y_vector":raw_train_target}

