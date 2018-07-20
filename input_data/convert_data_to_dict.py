import os, pickle,torchvision,torch, numpy
import pandas as pd
import torchvision.transforms as transforms
from sklearn import preprocessing
def get_data_dict(dataset_name,standardize_predictor=True):
    permissible_vals = ["pima_indian","boston","subset_mnist","subset_cifar10","logistic_mnist","logistic_cifar10","logistic_8x8mnist"]
    permissible_vals += ["australian","german","heart","diabetes","breast","8x8mnist","sp500","1-PL","mnist","cifar10"]

    assert dataset_name in permissible_vals

    out_datadict = {"input":None,"target":None}

    if dataset_name=="pima_indian":
        # classification
        address = os.environ["PYTHONPATH"] + "/input_data/pima_india.csv"
        df = pd.read_csv(address, header=0, sep=" ")
        dfm = df.values
        y = dfm[:, 8]
        X = dfm[:, 1:8]


    elif dataset_name=="boston":
        # regression
        from sklearn.datasets import load_boston
        boston = load_boston()
        X = boston["data"]
        y = boston["target"]

    elif dataset_name=="diabetes":
        # regression
        from sklearn.datasets import load_diabetes
        diabetes = load_diabetes()
        X = diabetes["data"]
        y = diabetes["target"]

    elif dataset_name=="breast":
        # classifcation
        from sklearn.datasets import load_breast_cancer
        breast = load_breast_cancer()
        X = breast["data"]
        y = breast["target"]

    elif dataset_name=="heart":
        # classification
        address = os.environ["PYTHONPATH"] + "/input_data/heart.csv"
        df = pd.read_csv(address, header=None, sep=" ")
        y = df[13]
        y = y.values
        y = (y-1)*1.0
        df = df.drop(13,1)
        df2 = pd.get_dummies(df,columns=[1,2,5,6,8,10,12])
        X = df2.values

    elif dataset_name=="german":
        # classification
        address = os.environ["PYTHONPATH"] + "/input_data/german.csv"
        df = pd.read_csv(address, header=None, sep=" ")
        y = df[20]
        y = y.values
        y = (y - 1)*1.0
        df = df.drop(20,1)
        df2 = pd.get_dummies(df,columns=[0,2,3,5,6,8,9,11,13,14,16,18,19])
        X = df2.values

    elif dataset_name=="australian":
        # classification
        address = os.environ["PYTHONPATH"] + "/input_data/australian.dat"
        df = pd.read_csv(address, header=None, sep=" ")
        y = df[14]
        df = df.drop(14, 1)
        # columns selected are categorical . convert to dummy variables
        df2 = pd.get_dummies(df,columns=[0,3,4,5,7,8,10,11])
        X = df2.values
        y = y.values

    elif dataset_name=="subset_mnist":
        abs_address = os.environ["PYTHONPATH"] + "/input_data/subset_mnist.pkl"
        # first check if data exists
        if os.path.isfile(abs_address):
            data_dict = pickle.load(open(abs_address, 'rb'))
            X = data_dict["input"]
            y= data_dict["target"]
        else:
            abs_address = os.environ["PYTHONPATH"] + "/data/"


            train_dataset = torchvision.datasets.MNIST(root=abs_address,
                                                       train=True,
                                                       transform=transforms.ToTensor(),
                                                       download=True)

            train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                       batch_size=60000,
                                                       shuffle=True)
            for i, (images, labels) in enumerate(train_loader):
                raw_train_X, raw_train_target = images, labels

            dataset = {"input": raw_train_X, "target": raw_train_target}
            subset_data = subset_dataset(dataset, 10, 23)
            abs_address = os.environ["PYTHONPATH"] + "/input_data/subset_mnist.pkl"
            with open(abs_address, 'wb') as f:
                 pickle.dump(subset_data, f)

            X = subset_data["input"]
            y = subset_data["target"]
    elif dataset_name=="mnist":

        # first check if data exists
        abs_address = os.environ["PYTHONPATH"] + "/data/"
        train_dataset = torchvision.datasets.MNIST(root=abs_address,
                                                   train=True,
                                                   transform=transforms.ToTensor(),
                                                   download=True)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=60000,
                                                   shuffle=True)
        for i, (images, labels) in enumerate(train_loader):
            raw_train_X, raw_train_target = images, labels

        dataset = {"input": raw_train_X, "target": raw_train_target}


        X = (raw_train_X.contiguous().view(60000,28*28)).numpy()
        y = (raw_train_target).numpy()



    elif dataset_name == "cifar10":

        # first check if data exists
        abs_address = os.environ["PYTHONPATH"] + "/data/"
        train_dataset = torchvision.datasets.CIFAR10(root=abs_address,
                                                   train=True,
                                                   transform=transforms.ToTensor(),
                                                   download=True)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=60000,
                                                   shuffle=True)
        for i, (images, labels) in enumerate(train_loader):
            raw_train_X, raw_train_target = images, labels

        dataset = {"input": raw_train_X, "target": raw_train_target}

        X = raw_train_X[:,0,:,:]
        X = (X.contiguous().view(50000,32*32)).numpy()
        y = raw_train_target.numpy()

    elif dataset_name == "logistic_mnist":
        subset_mnist_dict = get_data_dict("subset_mnist")
        y = subset_mnist_dict["target"]
        num_classes = len(numpy.unique(y))
        index_to_choose_from = [index for index in range(len(y)) if y[index] == 1 or y[index]==0]
        X = subset_mnist_dict["input"][index_to_choose_from,:]
        y = y[index_to_choose_from]
    elif dataset_name == "subset_cifar10":
        abs_address = os.environ["PYTHONPATH"] + "/input_data/subset_cifar10.pkl"
        # first check if data exists
        if os.path.isfile(abs_address):
            data_dict = pickle.load(open(abs_address, 'rb'))
            X = data_dict["input"]
            y = data_dict["target"]
        else:
            abs_address = os.environ["PYTHONPATH"] + "/data/"

            train_dataset = torchvision.datasets.CIFAR10(root=abs_address,
                                                       train=True,
                                                       transform=transforms.ToTensor(),
                                                       download=True)

            train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                       batch_size=60000,
                                                       shuffle=True)
            for i, (images, labels) in enumerate(train_loader):
                raw_train_X, raw_train_target = images, labels

            dataset = {"input": raw_train_X, "target": raw_train_target}
            subset_data = subset_dataset(dataset=dataset, size_per_class=10, seed=23)
            address = os.environ["PYTHONPATH"] + "/input_data/subset_cifar10.pkl"
            with open(address, 'wb') as f:
                pickle.dump(subset_data, f)

            X = subset_data["input"]
            y = subset_data["target"]




    elif dataset_name == "logistic_cifar10":
        subset_dict = get_data_dict("subset_cifar10")
        y = subset_dict["target"]
        num_classes = len(numpy.unique(y))
        index_to_choose_from = [index for index in range(len(y)) if y[index] == 1 or y[index]==0]
        X = subset_dict["input"][index_to_choose_from,:]
        y = y[index_to_choose_from]


    elif dataset_name == "8x8mnist":
        from sklearn.datasets import load_digits
        digits = load_digits()
        X = digits["data"]
        y = digits["target"]


    elif dataset_name =="logistic_8x8mnist":
        subset_mnist_dict = get_data_dict("8x8mnist")
        y = subset_mnist_dict["target"]
        num_classes = len(numpy.unique(y))
        index_to_choose_from = [index for index in range(len(y)) if y[index] == 1 or y[index] == 0]
        X = subset_mnist_dict["input"][index_to_choose_from,:]
        y = y[index_to_choose_from]

    elif dataset_name =="sp500":
        address = os.environ["PYTHONPATH"] + "/input_data/sp500.csv"
        df = pd.read_csv(address, header=None, sep=" ")
        X = None
        y = df[0].values
        standardize_predictor = False

    elif dataset_name=="1-PL":
        numpy.random.seed(10)
        num_students = 50
        true_theta  = numpy.random.randn(1)*numpy.sqrt(10)
        true_b = numpy.random.randn(num_students)*numpy.sqrt(10)
        prob = 1/(1+numpy.exp(-(true_b-true_theta)))
        print(prob)
        y = numpy.zeros(num_students)
        for i in range(num_students):
            y[i] = numpy.random.binomial(1,prob[i],1)

        X = None


    if standardize_predictor:
        X = preprocessing.scale(X)
    out_datadict.update({"input": X, "target": y})

    return(out_datadict)


def subset_dataset(dataset,size_per_class,seed=1):
    # dataset
    numpy.random.seed(seed)
    X = dataset["input"]
    y= dataset["target"]
    num_classes = len(numpy.unique(y))
    final_indices = []
    for i in range(num_classes):
        index_to_choose_from = [index for index in range(len(y)) if y[index]==i]
        assert len(index_to_choose_from)>=size_per_class
        final_indices+=numpy.random.choice(index_to_choose_from,size=size_per_class,replace=False).tolist()

    outX = X[final_indices,:]
    outy = y[final_indices]

    outX = outX[:,0,:,:]
    outX = outX.contiguous().view(outX.shape[0],-1)


    out = {"input":outX.numpy(),"target":outy.numpy()}

    return(out)



def subset_data_dict(dataset_dict,subset_size):
    original_size = dataset_dict["input"].shape[0]
    subset_indices= numpy.random.choice(list(range(original_size)), size=subset_size, replace=False)
    out = {"input":dataset_dict["input"][subset_indices,:],"target":dataset_dict["target"][subset_indices]}
    return(out)
#out = get_data_dict("1-PL",standardize_predictor=False)


#print(out["target"])