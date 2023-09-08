# Test 1
import numpy as np
def train_test_split(x, y, test_size=0.2):

    # Split data
    num_samples = len(x)
    num_test = int(test_size * num_samples)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    test_indices = indices[:num_test]
    train_indices = indices[num_test:]

    x_train = [x[i] for i in train_indices]
    x_test = [x[i] for i in test_indices]

    y_train = [y[i] for i in train_indices] 
    y_test = [y[i] for i in test_indices]

    return x_train, x_test, y_train, y_test
  

def generate_data(num_samples=100, dimension=1, test_size=0.2, m=7, b=3, low=0, high=10):

    # Generate x values
    x = np.random.uniform(low=low, high=high, size=(num_samples, dimension))

    # Compute y values with noise
    if dimension == 1:
        noise = np.random.normal(loc=0, scale=1, size=num_samples)
        y = (m * x) + b + noise
    
    else:
        noise = np.random.normal(loc=0, scale=1, size=num_samples)
        y = np.dot(x, np.array([m]*dimension)) + b + noise

    # Split data into train/test
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size)

    return xtrain, xtest, ytrain, ytest
# Generate data
# xtrain, xtest, ytrain, ytest = generate_data()

def q1():
    xtrain, xtest, ytrain, ytest = generate_data(num_samples=10000,test_size=0.2)
    xtrain, xtest, ytrain, ytest = np.array(xtrain, dtype=np.float32), np.array(xtest, dtype=np.float32), np.array(ytrain, dtype=np.float32), np.array(ytest, dtype=np.float32)
    print(xtrain.shape, xtest.shape, ytrain.shape, ytest.shape)
    return {
        "train": (xtrain, ytrain),
        "test": (xtest, ytest)
    }


def q2():
    xtrain, xtest, ytrain, ytest = generate_data(num_samples=10000, dimension=2, test_size=0.2, low=-10, high=10)
    xtrain, xtest, ytrain, ytest = np.array(xtrain, dtype=np.float32), np.array(xtest, dtype=np.float32), np.array(ytrain, dtype=np.float32), np.array(ytest, dtype=np.float32)
    print(xtrain.shape, xtest.shape, ytrain.shape, ytest.shape)
    print(xtrain)
    print(ytrain)
    # num_samples = 1000
    # x = np.random.uniform(-10, 10, size=(num_samples, 2))
    # noise = np.random.normal(0, 1, num_samples)
    # y = 3 * x[:, 0] + 4 * x[:, 1] + 5 + noise
    # print(x.shape, y.shape)
    # # pass

def q2b():
    xtrain, xtest, ytrain, ytest = generate_data(num_samples=10000, dimension=3, test_size=0.2, low=0, high=10)
    xtrain, xtest, ytrain, ytest = np.array(xtrain, dtype=np.float32), np.array(xtest, dtype=np.float32), np.array(ytrain, dtype=np.float32), np.array(ytest, dtype=np.float32)
    print(xtrain.shape, xtest.shape, ytrain.shape, ytest.shape)
    print(xtrain)
    print(ytrain)

def q3():
    pass

def q4():
    pass

def q5():
    pass

def q6():
    pass



q2b()