import numpy as np

def poly_train(w, DataX, DataY):
    Mp = len(w)
    N = len(DataX)

    Phi = np.vstack((np.ones((1, N)),
                     np.tile(DataX, (Mp - 1, 1))
                     ))

    Phi = np.cumprod(Phi, axis=0)
    A = np.matmul(Phi, np.transpose(Phi))
    b = np.matmul(Phi, np.transpose(DataY))

    w = np.linalg.solve(A, b)

    return w


def test_sc():
    DatX = np.array([0, 0.111, 0.222, 0.333, 0.444, 0.556, 0.667, 0.778, 0.889, 1])
    DatY = np.array([-0.028, 0.988, 1.387, 1.625, 1.089, 0.713, 0.328, 0.535, 1.112, 2.004])

    w = poly_train([0, 0, 0, 0, 0, 0], DatX, DatY)

    print('w = ', w)


if __name__ == '__main__':
    print('test')
    test_sc()


