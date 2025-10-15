import numpy as np

def linear_train(DataX, DataY):
    '''
    DataX: [N,] or [N, 1] input vector
    DataY: [N,] or [N, 1] output vector
    Returns: w = [intercept, slope]
    '''
    N = len(DataX)
    
    Phi = np.vstack((np.ones(N), DataX)).T 
    w = np.linalg.inv(Phi.T @ Phi) @ (Phi.T @ DataY)
    
    return w 


def test_linear():
    DataX = np.linspace(0, 10, 20)         
    DataY = 2 * DataX + 1               

    w = linear_train(DataX, DataY)
    
    print('Linear regression coefficients:')
    print(f'Intercept: {w[0]}')  
    print(f'Slope: {w[1]}')     
    
if __name__ == '__main__':
    test_linear()
