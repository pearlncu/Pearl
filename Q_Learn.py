import numpy as np

R = np.array([
    [-1, -1, -1, -1, 0, -1],
    [-1, -1, -1, 0, -1, 100],
    [-1, -1, -1, 0, -1, -1],
    [-1, 0, 0, -1, 0, -1],
    [0, -1, -1, 0, -1, 100],
    [-1, 0, -1, -1, 0, 100]
])

Q = np.zeros((6, 6))

def dumpmat(label, matrix): #print matrix
    print(label)
    for row in matrix:
        for val in row:
            print(f"{val:5} ", end='')
        print()

def calc(R, Q): 
    alpha = 0.8
    Q1 = np.copy(Q)  # Create a new Q-value matrix to avoid updating in-place
    
    for i in range(5):
        for j in range(6):
            if R[i, j] != -1:
                max_val = np.max(Q[j, :])
                Q1[i, j] = R[i, j] + alpha * max_val

    return Q1

dumpmat("Reward Matrix", R)
dumpmat("Initial Q Matrix", Q)

for i in range(7):  # You can adjust the number of iterations
    Q = calc(R, Q)

    dumpmat("Final Q Matrix"+str(i), Q)