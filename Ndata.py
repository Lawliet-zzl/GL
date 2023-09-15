import pandas as pd
import numpy as np
import torch
import csv
import os
from torch.utils.data import Dataset

class LoadDataset(Dataset):
	"""docstring for LoadDataset"""
	def __init__(self,  root):
		super(LoadDataset, self).__init__()
		self.filename = root
		self._parse_list()

	def _parse_list(self):
		self.data_list = case_train=np.loadtxt(self.filename,delimiter=',',skiprows=0)
		# self.data_list = [x for x in open(self.filename)]

	def __getitem__(self, index):
		record = self.data_list[index]
		#torch.Tensor(train_X)
		return torch.Tensor(record[0:len(record)-1]), torch.Tensor(np.array(record[-1])).long()

	def __len__(self):
		return len(self.data_list)

def data_to_tensor(train_X, test_X, train_y, test_y):
    train_X = torch.Tensor(train_X)
    test_X = torch.Tensor(test_X)
    train_y = torch.Tensor(train_y.astype(float)).long()
    test_y = torch.Tensor(test_y.astype(float)).long()
    return train_X, test_X, train_y, test_y

def load_iris():
    dataset = pd.read_csv('dataset/iris.csv')
    dataset.loc[dataset.species=='Iris-setosa', 'species'] = 0
    dataset.loc[dataset.species=='Iris-versicolor', 'species'] = 1
    dataset.loc[dataset.species=='Iris-virginica', 'species'] = 2
    X_dim = dataset.shape[1] - 1
    y_dim = 3
    X = dataset[dataset.columns[0:X_dim]].values
    y = dataset.species.values
    # train_X, test_X, train_y, test_y = train_test_split(dataset[dataset.columns[0:X_dim]].values, dataset.species.values, test_size=rho)
    return X, y

def load_phishing():
    dataset = pd.read_csv('dataset/phishing.csv')
    dataset.loc[dataset.species==-1, 'species'] = 0
    dataset.loc[dataset.species==1, 'species'] = 1
    X_dim = dataset.shape[1] - 1
    y_dim = 2
    X = dataset[dataset.columns[0:X_dim]].values
    y = dataset.species.values
    return X, y

def load_htru2():
    dataset = pd.read_csv('dataset/htru2.csv')
    X_dim = dataset.shape[1] - 1
    y_dim = 2
    X = dataset[dataset.columns[0:X_dim]].values
    y = dataset[dataset.columns[8]].values
    return X, y

def load_abalone():
    dataset = pd.read_csv('dataset/abalone.csv')
    dataset.loc[dataset.species=='M', 'species'] = 0
    dataset.loc[dataset.species=='F', 'species'] = 1
    dataset.loc[dataset.species=='I', 'species'] = 2
    X_dim = dataset.shape[1] - 1
    y_dim = 3
    X = dataset[dataset.columns[0:X_dim]].values
    y = dataset.species.values
    return X, y

def load_arcene():
    dataset = pd.read_csv('dataset/arcene.csv')
    X_dim = dataset.shape[1] - 1
    y_dim = 2
    X = dataset[dataset.columns[0:X_dim]].values
    y = dataset[dataset.columns[X_dim]].values + 1
    for i in range(len(y)):
        if y[i] == 2:
            y[i] = 1
    return X, y

def load_wdbc():
    dataset = pd.read_csv('dataset/wdbc.csv')
    map2values(dataset, 'species')
    X_dim = dataset.shape[1] - 1
    y_dim = 2
    X = dataset[dataset.columns[0:X_dim]].values
    y = dataset.species.values
    return X, y

def load_covid19():
    dataset = pd.read_csv('dataset/covid19.csv')
    map2values(dataset, "Country")
    X = dataset.drop(['Severity_Mild','Severity_Moderate','Severity_None','Severity_Severe'],axis=1)
    X = pd.concat([X,pd.get_dummies(dataset["Country"])], axis = 1).values

    y = ((dataset['Severity_None'] 
        + 2*dataset['Severity_Mild'] 
        + 3*dataset['Severity_Moderate'] 
        + 4*dataset['Severity_Severe']) - 1).values
    # print(X.shape, len(pd.unique(y)))
    return X, y

def load_gisette():
    dataset = pd.read_csv('dataset/gisette.csv')
    map2values(dataset, "target")
    
    X_dim = dataset.shape[1] - 1
    X = dataset[dataset.columns[0:X_dim]].values
    y = dataset['target'].values
    return X, y

def load_wine():
    dataset = pd.read_csv('dataset/WineQT.csv')
    map2values(dataset, "quality")
    
    X = dataset[dataset.columns[0:11]].values
    y = dataset['quality'].values
    y[y >= 3] = 3
    return X, y

def load_skyserver():
    dataset = pd.read_csv('dataset/Skyserver.csv')
    map2values(dataset, "class")
    
    # dataset = dataset.drop(['objid', 'rerun', 'field', 'specobjid', 'plate' ,'fiberid', 'mjd', 'camcol'],axis=1)

    X_dim = dataset.shape[1] - 1
    X = dataset[dataset.columns[0:X_dim]].values
    y = dataset['class'].values
    # print(X.shape, len(pd.unique(y)))
    # print(X,pd.unique(y))
    # print(pd.value_counts(y))
    return X, y

def load_stellar():
    dataset = pd.read_csv('dataset/stellar.csv')
    map2values(dataset, "class")
    
    X_dim = dataset.shape[1] - 1
    X = dataset[dataset.columns[0:X_dim]].values
    y = dataset['class'].values
    return X, y


def load_speech():
    dataset = pd.read_csv('dataset/Speech.csv')
    map2values(dataset, "language")
    
    X_dim = dataset.shape[1] - 1
    X = dataset[dataset.columns[0:X_dim]].values
    y = dataset['language'].values
    # print(X.shape, len(pd.unique(y)))
    # print(X,pd.unique(y))
    # print(pd.value_counts(y))
    return X, y

def load_grid():
    dataset = pd.read_csv('dataset/grid.csv')
    map2values(dataset, "stabf")
    
    X_dim = dataset.shape[1] - 1
    X = dataset[dataset.columns[0:X_dim]].values
    y = dataset['stabf'].values
    return X, y

def load_SHABD():
    dataset = pd.read_csv('dataset/SHABDtrain(grayscale).csv')
    # print(pd.unique(dataset.loc[:,'label']))
    map2values(dataset, "label")

    X = dataset[dataset.columns[2:dataset.shape[1]]].values
    y = dataset['label'].values
    return X, y

def load_bank():
    dataset = pd.read_csv('dataset/bank.csv')
    map2values(dataset, "y")
    X = dataset.loc[:, ["age", "pdays","previous","emp.var.rate","cons.price.idx","cons.conf.idx","euribor3m","nr.employed"]]
    F = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "day_of_week", "poutcome"]
    for f in F:
        b = pd.get_dummies(dataset[f])
        X = pd.concat([X,b], axis = 1)
    X = X[X.columns[0:X.shape[1]]].values
    y = dataset['y'].values
    return X, y

def load_IBM():
    dataset = pd.read_csv('dataset/IBM.csv')

    X = dataset.loc[:, ["Age", "DistanceFromHome", "Education", "EnvironmentSatisfaction", "JobSatisfaction", "MonthlyIncome", "NumCompaniesWorked", "WorkLifeBalance"]]
    F = ["Attrition", "Department", "EducationField", "MaritalStatus"]
    for f in F:
        b = pd.get_dummies(dataset[f])
        X = pd.concat([X,b], axis = 1)
    X = X[X.columns[0:X.shape[1]]].values
    index = (dataset['YearsAtCompany'] > 20)
    dataset.loc[index,'YearsAtCompany'] = 21
    y = dataset['YearsAtCompany'].values
    # print(X.shape, len(pd.unique(y)))
    return X, y

def load_gene():
    X = pd.read_csv('dataset/gene/data.csv')
    y = pd.read_csv('dataset/gene/labels.csv')

    map2values(y, "Class")
    X = X[X.columns[1:X.shape[1]]].values
    y = y['Class'].values
    # print(X.shape, len(pd.unique(y)))
    return X, y

def load_arrhythmia():
    dataset = pd.read_csv('dataset/Arrhythmia.csv')
    # print(pd.unique(dataset.loc[:,'label']))
    # map2values(dataset, "label")

    X_dim = dataset.shape[1] - 1
    X = dataset[dataset.columns[0:X_dim]].values
    y = dataset[dataset.columns[X_dim]].values
    # print(X.shape, len(pd.unique(y)))
    return X, y


def write_data(X, y, name, OOD, dtype):
    filename = "dataset2/" + name + "_" + str(OOD) + "_" + dtype + ".csv"
    with open(filename, 'w') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        for i in range(len(X)):
        	row = X[i].tolist()
        	row.append(y[i])
        	logwriter.writerow(row)

def map2values(dataset, index):
    D = pd.unique(dataset[index]).tolist()
    def featuremap(x):
        return D.index(x)
    dataset[index] = dataset[index].map(featuremap)

def feature_normalize(train_ID_X, test_ID_X, test_OOD_X):
    mu = np.mean(train_ID_X,axis=0)
    std = np.std(train_ID_X,axis=0)
    train_ID_X, test_ID_X, test_OOD_X = (train_ID_X - mu)/std, (test_ID_X - mu)/std, (test_OOD_X - mu)/std
    remove_nan(train_ID_X)
    remove_nan(test_ID_X)
    remove_nan(test_OOD_X)
    return train_ID_X, test_ID_X, test_OOD_X

def remove_nan(data):
    data[np.isnan(data)] = 0
    data[np.isinf(data)] = 0

def ID_OOD_split(X, y, ood_label = 0):
    index_ID = np.where(y != ood_label)
    index_OOD = np.where(y == ood_label)

    ID_X, ID_y = X[index_ID], y[index_ID]
    OOD_X, OOD_y = X[index_OOD], y[index_OOD] - ood_label - 1

    for i in range(len(ID_X)):
        if ID_y[i] > ood_label:
            ID_y[i] = ID_y[i] - 1
    return ID_X, ID_y, OOD_X, OOD_y

def load_dim(dataname):
    if dataname == 'iris':
        num_sample, X_dim, y_dim, OOD = 150, 4, 3, 0
    elif dataname == 'htru2':
        num_sample, X_dim, y_dim, OOD = 17897, 8, 2, 0
    elif dataname == 'arcene':
        num_sample, X_dim, y_dim, OOD = 100, 10000, 2, 0
    elif dataname == "abalone":
        num_sample, X_dim, y_dim, OOD = 4177, 8, 3, 0
    elif dataname == "phishing":
        num_sample, X_dim, y_dim, OOD = 11055, 30, 2, 0
    elif dataname == "wdbc":
        num_sample, X_dim, y_dim, OOD = 569, 30, 2, 0
    elif dataname == 'wine':
        num_sample, X_dim, y_dim, OOD = 1143, 11, 4, 2
    elif dataname == 'covid19':
        num_sample, X_dim, y_dim, OOD = 316800, 33, 4, 2
    elif dataname == 'gisette':
        num_sample, X_dim, y_dim, OOD = 6000, 5000, 2, 0
    elif dataname == 'skyserver':
        num_sample, X_dim, y_dim, OOD = 100000, 17, 3, 1
        # num_sample, X_dim, y_dim, OOD = 100000, 9, 3, 2
    elif dataname == 'grid':
        num_sample, X_dim, y_dim, OOD = 60000, 13, 2, 0
    elif dataname == 'bank':
        num_sample, X_dim, y_dim, OOD = 41188, 103, 2, 0
    elif dataname == 'SHABD':
        num_sample, X_dim, y_dim, OOD = 243456, 1024, 384, 0
    elif dataname == 'IBM':
        num_sample, X_dim, y_dim, OOD = 1470, 24, 22, 21
    elif dataname == 'gene':
        num_sample, X_dim, y_dim, OOD = 801, 20531, 5, 4
    elif dataname == 'arrhythmia':
        num_sample, X_dim, y_dim, OOD = 87553, 187, 5, 3
    elif dataname == 'speech':
        num_sample, X_dim, y_dim, OOD = 3960, 12, 6, 2
    elif dataname == 'stellar':
        num_sample, X_dim, y_dim, OOD = 100000, 16, 3, 2
    else:
        num_sample, X_dim, y_dim = 0,0,0,0
    return num_sample, X_dim, y_dim, OOD

def load_data(data):
    num_sample, X_dim, y_dim, OOD = load_dim(data)
    if data == "iris":
        X, y = load_iris()
    elif data == "phishing":
        X, y = load_phishing()
    elif data == "htru2":
        X, y = load_htru2()
    elif data == "abalone":
        X, y = load_abalone()
    elif data == "arcene":
        X, y = load_arcene()
    elif data == "wdbc":
        X, y = load_wdbc()
    elif data == "wine":
        X, y = load_wine()
    elif data == "covid19":
        X, y = load_covid19()
    elif data == "gisette":
        X, y = load_gisette()
    elif data == "skyserver":
        X, y = load_skyserver()
    elif data == "grid":
        X, y = load_grid()
    elif data == "bank":
        X, y = load_bank()
    elif data == "SHABD":
        X, y = load_SHABD()
    elif data == "IBM":
        X, y = load_IBM()
    elif data == "gene":
        X, y = load_gene()
    elif data == "arrhythmia":
        X, y = load_arrhythmia()
    elif data == "speech":
        X, y = load_speech()
    elif data == "stellar":
        X, y = load_stellar()
    else:
        X, y = load_iris()

    return X, y, X_dim, y_dim, OOD

def get_loader_IOD(dataset, OOD, batch_size_ID = 128, batch_size_OOD = 128):
    num_sample, X_dim, y_dim, OOD_index = load_dim(dataset)

    if OOD == -1:
        OOD = OOD_index

    trainset = LoadDataset(root="dataset2/" + dataset + "_" + str(OOD) + "_train_ID.csv")
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_ID, shuffle=True)

    testset = LoadDataset(root="dataset2/" + dataset + "_" + str(OOD) + "_test_ID.csv")
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size_ID, shuffle=False)

    OODset = LoadDataset(root="dataset2/" + dataset + "_" + str(OOD) + "_test_OOD.csv")
    OODloader = torch.utils.data.DataLoader(OODset, batch_size=batch_size_OOD, shuffle=False)

    if y_dim == 2:
        y_dim = 3
    return trainloader, testloader, OODloader, X_dim, y_dim - 1