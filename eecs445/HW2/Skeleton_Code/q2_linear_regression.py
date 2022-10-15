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
    #np.apply_along_axis(np.dot(), )
    for i in size: 
        sum += (y[i] - (np.dot(theta, X[i,:]))) ** 2 #[i,:] returns entire row
        sum /= 2
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
    n = len(X)
    Phi = np.zeros((n, M+1))
    X_arr = []
    for i in X:
        for j in range(M+1):
            X_arr.append(X^j)
    Phi.append(X_arr)
    return Phi

def calculate_RMS_Error(X, y, theta):
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
    PhiX = generate_polynomial_features(X, M = 2)
    for i in range(len(X)):
        sum = np.dot(theta, PhiX)
        sum = (E_theta - y[i]) ** 2
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
    while (num_iterations <= 1e6) and (abs(new_loss - prev_loss) <= 1e-10):
        theta = theta - learning_rate*calculate_empirical_risk(X, y, theta)
        new_loss = calculate_empirical_risk(X, y, theta)
        num_iterations += 1
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

    while (num_iterations <= 1e6) and (abs(new_loss - prev_loss) <= 1e-10):
        curr_emp = (y[i] - theta * X[i, :]) ** 2 #empirical risk
        curr_emp /= 2
        theta = theta - learning_rate * curr_emp
        i += 1
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
    #(X transposed x X)^-1
    X_transpose = np.transpose(X)
    Xsquare = np.matmul(X_transpose, X) 
    XInvX = np.linalg.inv(Xsquare)

    #X transposed x y
    Xy = np.matmul(X_transpose, y)

    #Theta optimal
    theta = np.matmul(XInvX, Xy)
    return theta


def part_2_1(fname_train):
    # TODO: This function should contain all the code you implement to complete 2.1. Please feel free to add more plot commands.
    print("========== Part 2.1 ==========")

    X_train, y_train = load_data(fname_train)
    step_size = [1e-4, 1e-3, 1e-2, 1e-1]

    PhiX = generate_polynomial_features(X_train, M = 2)
    for i in step_size: #Step through different step sizes
        print("step_size =", i )
        GD_theta = ls_gradient_descent(X_train, y_train, learning_rate = i)
        GD_theta = np.dot(GD_theta, PhiX) #Multiplies GD theta with phi(x)
        print("Gradient Descent =", GD_theta)

        SGD_theta = ls_stochastic_gradient_descent(X_train, y_train, learning_rate = i)
        SGD_theta = np.dot(SGD_theta, PhiX) #Multiplies SGD theta with phi(x)
        print("Stochastic Gradient Descent =", SGD_theta)

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
