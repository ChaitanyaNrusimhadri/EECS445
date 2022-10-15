"""
EECS 445 - Introduction to Maching Learning
HW2 Q2 Linear Regression Optimization Methods)
~~~~~~
Follow the instructions in the homework to complete the assignment.
"""

import numpy as np
import matplotlib.pyplot as plt
from helper import load_data

def calculate_empirical_risk(X, y, theta):
    # TODO: Implement this function
    sum = 0
    size = np.size(y)
    for i in range(size): 
        sum += ((y[i] - (np.dot(theta, X[i,:]))) ** 2)/2 #[i,:] returns entire row
    risk = sum/size
    return risk


def generate_polynomial_features(X, M):
    """
    Create a polynomial feature mapping from input examples. Each element x
    in X is mapped to an (M+1)-dimensional polynomial feature vector 
    i.e. [1, x, x^2, ...,x^M].

    Args:
        X: np.array, shape (n, 1). Each row is one instance.
        M: a non-negative integer
    
    Returns:
        Phi: np.array, shape (n, M+1)
    """
    # TODO: Implement this function
    n = X.shape[0]
    Phi = np.zeros((n, M+1))
    #X_arr = []
    for i in range(X.shape[0]):
        for j in range(M+1):
            Phi[i][j] = X[i]**j
    #Phi.append(X_arr)
    return Phi

def calculate_RMS_Error(X, y, theta, M = 0):
    """
    Args:
        X: np.array, shape (n, d) 
        y: np.array, shape (n,)
        theta: np.array, shape (d,). Specifies an (d-1)^th degree polynomial

    Returns:
        E_rms: float. The root mean square error as defined in the assignment.
    """
    # TODO: Implement this function
    import math as mth
    E_rms = 0
    E_theta = 0
    sum = 0
    PhiX = generate_polynomial_features(X, M)
    for i in range(len(X)):
        sum = np.dot(theta, PhiX[i])
        sum = (sum - y[i]) ** 2
        E_theta += sum
    E_rms = mth.sqrt(E_theta/len(X))
    return E_rms


def ls_gradient_descent(X, y, learning_rate=0):
    """
    Implements the Gradient Descent (GD) algorithm for least squares regression.
    Note:
        - Please use the stopping criteria: number of iterations >= 1e6 or |new_loss - prev_loss| <= 1e-10

    Args:
        X: np.array, shape (n, d) 
        y: np.array, shape (n,)
    
    Returns:
        theta: np.array, shape (d,)
    """
    # TODO: Implement this function
    d = len(X[0])
    theta = np.zeros((d,))
    prev_loss = calculate_empirical_risk(X, y, theta)
    new_loss = 0
    #step_size = [1e-4, 1e-3, 1e-2, 1e-1]
    num_iterations = 0
    while (num_iterations <= 1e6) and (abs(new_loss - prev_loss) >= 1e-10):
        grad_sum = 0
        for i in range(X.shape[0]):
            grad_sum += (y[i] - np.dot(theta, X[i])) * (-X[i])
        theta = theta - learning_rate * grad_sum/X.shape[0]
        prev_loss = new_loss
        new_loss = calculate_empirical_risk(X, y, theta)
        num_iterations += 1
    print("iterations = ", num_iterations)
    return theta


def ls_stochastic_gradient_descent(X, y, learning_rate=0):
    """
    Implements the Stochastic Gradient Descent (SGD) algorithm for least squares regression.
    Note:
        - Please do not shuffle your data points.
        - Please use the stopping criteria: number of iterations >= 1e6 or |new_loss - prev_loss| <= 1e-10
    
    Args:
        X: np.array, shape (n, d) 
        y: np.array, shape (n,)
    
    Returns:
        theta: np.array, shape (d,)
    """
    # TODO: Implement this function
    d = len(X[0])
    theta = np.zeros((d,))
    prev_loss = calculate_empirical_risk(X, y, theta)
    new_loss = 0
    step_size = [1e-4, 1e-3, 1e-2, 1e-1]
    num_iterations = 0

    i = 0 #used for accessing current i in X and y matrices during SGD

    while (num_iterations <= 1e6) and (abs(new_loss - prev_loss) >= 1e-10):
        for i in range(X.shape[0]):
            curr_emp = (y[i] - np.dot(theta, X[i])) * (-X[i])  #empirical risk
            theta = theta - learning_rate * curr_emp
            num_iterations += 1
        prev_loss = new_loss
        new_loss = calculate_empirical_risk(X,y, theta)
    print("num_it =", num_iterations)
    return theta


def closed_form_optimization(X, y, reg_param=0):
    """
    Implements the closed form solution for least squares regression.

    Args:
        X: np.array, shape (n, d) 
        y: np.array, shape (n,)
        `reg_param`: float, an optional regularization parameter

    Returns:
        theta: np.array, shape (d,)
    """
    # TODO: Implement this function
    d = len(X[0])
    theta = np.zeros((d,))

    """
    #(X transposed x X)^-1
    X_transpose = np.transpose(X)
    Xsquare = np.matmul(X_transpose, X) 
    XInvX = np.linalg.inv(Xsquare)

    #X transposed x y
    Xy = np.matmul(X_transpose, y)

    #Theta optimal
    theta = np.matmul(XInvX, Xy)
    """

    #Create identity matrix size d x d and multiply reg_param(lambda) to it 
    #X^T times X is a d x d matrix since X is a n x d matrix
    id_matrix = np.identity(d)
    lambda_matrix = reg_param * id_matrix

    #Multiplies X^T to X
    X_transpose = np.transpose(X)
    Xsquare = np.matmul(X_transpose, X) 

    #X transposed times y
    Xy = np.matmul(X_transpose, y)

    #Creates added matrix and inverses it 
    X_add_lambda = lambda_matrix + Xsquare
    InvX = np.linalg.inv(X_add_lambda)

    #Multiplies inversed matrix with X^T*y
    theta = np.matmul(InvX, Xy)

    return theta


def part_2_1(fname_train):
    # TODO: This function should contain all the code you implement to complete 2.1. Please feel free to add more plot commands.
    print("========== Part 2.1 ==========")

    X_train, y_train = load_data(fname_train)
    step_size = [1e-4, 1e-3, 1e-2, 1e-1]

    import timeit

    PhiX = generate_polynomial_features(X_train, M = 1)
    """
    for i in step_size: #Step through different step sizes
        start = timeit.default_timer()
        print("step_size =", i )
        GD_theta = ls_gradient_descent(PhiX, y_train, learning_rate = i)
        print("Gradient Descent =", GD_theta)
        stop = timeit.default_timer()
        print('GD Time: ', stop - start)  


        print("----------------")
        start = timeit.default_timer()
        SGD_theta = ls_stochastic_gradient_descent(PhiX, y_train, learning_rate = i)
        #SGD_theta = np.dot(SGD_theta, PhiX) #Multiplies SGD theta with phi(x)
        print("Stochastic Gradient Descent =", SGD_theta)
        stop = timeit.default_timer()
        print('SGD Time: ', stop - start) 
        print("-----------------------------------------------------")
    """
    start = timeit.default_timer()
    closed_form = closed_form_optimization(PhiX, y_train)
    print("closed form optimization =", closed_form)
    stop = timeit.default_timer()
    print('Closed Form Optimization Time: ', stop - start)

    print("Done!")
    plt.plot(X_train, y_train, 'ro')
    plt.legend()
    plt.savefig('q2_1.png', dpi=200)
    plt.close()


def part_2_2(fname_train, fname_validation):
    # TODO: This function should contain all the code you implement to complete 2.2
    print("=========== Part 2.2 ==========")

    X_train, y_train = load_data(fname_train)
    X_validation, y_validation = load_data(fname_validation)

    # (a) OVERFITTING

    errors_train = np.zeros((11,))
    errors_validation = np.zeros((11,))
    # Add your code here

    
    for i in range(11):
        PhiX = generate_polynomial_features(X_train, M = i)
        theta = closed_form_optimization(PhiX, y_train)

        errors_train[i] = calculate_RMS_Error(X_train, y_train, theta, M = i)
        errors_validation[i] = calculate_RMS_Error(X_validation, y_validation, theta, M = i)
    

    plt.plot(errors_train,'-or',label='Train')
    plt.plot(errors_validation,'-ob', label='Validation')
    plt.xlabel('M')
    plt.ylabel('$E_{RMS}$')
    plt.title('Part 2.2.a')
    plt.legend(loc=1)
    plt.xticks(np.arange(0, 11, 1))
    plt.savefig('q2_2_a.png', dpi=200)
    plt.close()


    # (b) REGULARIZATION

    errors_train = np.zeros((10,))
    errors_validation = np.zeros((10,))
    L = np.append([0], 10.0 ** np.arange(-8, 1))
    # Add your code here

    PhiX = generate_polynomial_features(X_train, M = 10)

    lambda_arr = [0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

    for i in lambda_arr:
        theta = closed_form_optimization(PhiX, y_train, reg_param = i)
        errors_train[i] = calculate_RMS_Error(X_train, y_train, theta, M = i)
        errors_validation[i] = calculate_RMS_Error(X_validation, y_validation, theta, M = i)

    plt.figure()
    plt.plot(L, errors_train, '-or', label='Train')
    plt.plot(L, errors_validation, '-ob', label='Validation')
    plt.xscale('symlog', linthresh=1e-8)
    plt.xlabel('$\lambda$')
    plt.ylabel('$E_{RMS}$')
    plt.title('Part 2.2.b')
    plt.legend(loc=2)
    plt.savefig('q2_2_b.png', dpi=200)
    plt.close()

    print("Done!")


def main(fname_train, fname_validation):
    part_2_1(fname_train)
    part_2_2(fname_train, fname_validation)

if __name__ == '__main__':
    main("dataset/q2_train.csv", "dataset/q2_validation.csv")
