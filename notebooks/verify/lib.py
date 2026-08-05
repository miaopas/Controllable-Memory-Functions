import itertools
import numpy as np

def complex_random(size):
    np.random.seed(0)
    real_part = np.random.random(size)
    imag_part = np.random.random(size)
    return real_part + 1j * imag_part


def one_layer_recurrent(Y, W, V, x):
    T = len(x)
    k = len(W)
    y_iter = []
    ft = np.zeros((k,1), dtype=complex)
    for s in range(T):
        ft = Y @ ft + W*x[s]
        yt = V.T @ ft
        y_iter.append(yt.item())
    y_iter
    return np.array(y_iter)

def one_layer_closed_form(Y, W, V, x):
    y_form = []
    T = len(x)
    k = len(W)
    for t in range(T):
        total = 0.0
        for s in range(t+1):
            # proper matrix power and scalar indexing:
            total += (V.T @ np.linalg.matrix_power(Y, s) @ W).item() * x[t-s]
        y_form.append(total.item())
    return np.array(y_form)

def n_layer_recurrent(A, B, C, x):
    n, m = A.shape[0], A.shape[1]
    y_iter = []
    gt = np.zeros((n,m,1), dtype=complex)
    T = len(x)

    for s in range(T):
        gt[0] = A[0] @ gt[0] + B[0] * x[s]
        for i in range(1,n):
            gt[i] = A[i] @ gt[i] + B[i]@gt[i-1]
        yt = C.T @ gt[-1]
        y_iter.append(yt.item())
    return np.array(y_iter)

def n_layer_recurrent_closed_form(A, B, C, x):
    m = len(C) # hidden dim
    n = len(A) # num of layers
    T = len(x)

    y_form = []
    for t in range(T):
        total = 0.0
        for s in range(t+1):
            for i in itertools.product(range(m), repeat=n):
                value1 = [C[i[0]].item()]+ [ B[n-l-1][i[l],i[l+1]] for l in range(n-1)] + [B[0][i[-1]].item()]
                value1 = np.prod(value1)
                sum2 = 0
                for j in itertools.product(range(T), repeat=n):
                    if np.sum(j) != s:
                        continue
                    value2 = [A[l,i[n-l-1],i[n-l-1]]**j[n-l-1] for l in range(n)]
                    value2 = np.prod(value2)
                    sum2 += value2
                total += value1 * sum2* x[t-s]
        y_form.append(total.item())
    return np.array(y_form)


def get_n_layer_param(alpha,W,V, n,m, real=True):
    k = len(W)
    assert k == n*(m-1)+1

    alpha_divided = [np.array(alpha[:m])] + [np.append(alpha[m + i * (m - 1):m + (i + 1) * (m - 1)], 0) for i in range(n - 1)]

    A = []
    for a in alpha_divided:
        A.append(np.diag(a))
    A = np.array(A)

    

    if real:
        Z0 = (n*np.abs(V*W).max())**(1/(n+1))
    else:
        Z0 = 2*(np.abs(V*W).max())**(1/(n+1))


    Z_vec = V * W
    Z_list = Z_vec.flatten().tolist()  # make Z into a simple list of length k

    Z = [np.array(Z_list[:m])] + [
        np.append(Z_list[m + i * (m - 1) : m + (i + 1) * (m - 1)], 0)
        for i in range(n - 1)
    ]


    a = alpha_divided
    def H(x,y):
        total = 0
        for i in range(x, n):
            value1 = np.prod([a[i][y] - a[p][y] for p in range(x)])
            total += Z[i][y] * value1 * a[x][y] / a[i][y]**(x+1)
        return total

    C = Z0 * np.ones((m,1))
    B = [Z0 * np.ones((m,1))] + [np.zeros((m, m), dtype=complex) for _ in range(n-1)]

        # set values for B

    for s in range(1,n-1):
        for i in range(0,m-1):
            B[s][i,i] = Z0

    for s in range(2,n):
        B[s][m-1,m-1] = Z0

    B[1][m-1,m-1] = Z[0][m-1]/(Z0**n) 

    for i in range(m-1):
        B[n-1][i,i] = H(n-1,i)/(Z0**n) 

    for s in range(1,n):
        for i in range(m-1):
            B[s][m-1,i] = H(s-1,i)/(Z0**n)

    # for i in range(1,m):
    #     B [i] = B[i].T
    return A, B, C