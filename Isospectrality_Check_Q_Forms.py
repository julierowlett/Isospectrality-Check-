import numpy as np
import itertools
import sympy
import time

def quadratic_form_is_even(Q):
    if not np.all(np.abs(Q - np.round(Q)) < 0.0001): # Check if all entries are integers
        return False

    diagonal_elements = np.diag(Q) 
    for dia in diagonal_elements: # Check if all diagonal entries are even
        if dia % 2 > 0.0001 and 2 - (dia % 2) > 0.0001:
            return False

    return True

def generate_vectors(norm_lower_limit, norm_upper_limit, dimensions): # Generate all possible vectors bounded by a given norm
    vectors = []
    norm_upper_limit_int = int(np.ceil(norm_upper_limit))
    for values in itertools.product(range(-norm_upper_limit_int, norm_upper_limit_int + 1), repeat = dimensions):
        vector = np.array(values)
        norm = np.linalg.norm(vector)
        if norm_lower_limit <= norm and norm <= norm_upper_limit:
            vectors.append(vector)
    return vectors

def compute_NQ(Q):
    N_Q = 1
    Q_inv = np.linalg.inv(Q)
    while quadratic_form_is_even(N_Q*Q_inv) == False:
        N_Q += 1
    return N_Q

def compute_mu0(N):
    factors = sympy.factorint(N)
    result = 1.0
    for prime, exponent in factors.items():
        result *= (1 + 1 / prime)
    return int(np.ceil(N*result))

def compute_representation_number(Q, n):
    eigenvalues, _ = np.linalg.eig(Q)
    eigenvalues = np.real(eigenvalues)
    lambda_min = np.min(eigenvalues)
    lambda_max = np.max(eigenvalues)
    
    norm_lower_limit = np.sqrt(n/lambda_max)
    norm_upper_limit = np.sqrt(n/lambda_min)
    vectors = generate_vectors(norm_lower_limit, norm_upper_limit, len(Q))
    result = 0
    for v in vectors:
        if abs(np.dot(v.T, np.dot(Q, v)) - n) < 0.001:
            result += 1
    return result

def isospectral_lattices(A1, A2):
    Q1 = np.matmul(np.transpose(A1), A1)
    Q2 = np.matmul(np.transpose(A2), A2)
    return isospectral_quadratic_forms(Q1, Q2)

def isospectral_quadratic_forms(P, Q):
    if np.allclose(P, P.T) == False or np.allclose(Q, Q.T) == False:
        raise Exception("P, Q not symmetric")
        
    eigenvalues1, _ = np.linalg.eig(P)
    eigenvalues2, _ = np.linalg.eig(Q)
    lambda_min1 = np.min(eigenvalues1)
    lambda_min2 = np.min(eigenvalues2)
    
    if lambda_min1 <= 0 or lambda_min2 <= 0:
        raise Exception("Q1 and Q2 are not positive definite")
    
    if quadratic_form_is_even(P) == False or quadratic_form_is_even(Q) == False:
        raise Exception("P, Q not even")
    
    if len(P) != len(Q):
        return False
    
    if abs(np.linalg.det(P) - np.linalg.det(Q)) > 0.00001:
        print("det(P) = " + str(np.linalg.det(P)) + ", det(Q) = " + str(np.linalg.det(Q)))
        return False
    else:
        print("Same determinant: " + str(np.linalg.det(P)))
    N_P = compute_NQ(P)
    N_Q = compute_NQ(Q)
    if N_P != N_Q:
        print("N_P = " + str(N_P) + ", det(N_Q) = " + str(N_Q))
        return False
    else:
        print("Same N_P = " + str(N_P))
    
    last_coefficient = int(compute_mu0(N_P)*len(P)/24 + 1)*2
    print("last_coeffcient = " + str(last_coefficient))
    time.sleep(2)
    
    for i in range(0, last_coefficient + 1, 2): # Since even quadratic forms only take even values, we don't need to check the odd representation numbers
        r1 = compute_representation_number(P, i)
        r2 = compute_representation_number(Q, i)
        if r1 != r2:
            print("Different representation number at i = " + str(i) + ": " + str(compute_representation_number(P, i)) + " vs. " + str(compute_representation_number(Q, i)))
            return False
        else:
            print("Same representation number at i = " + str(i) + ": " + str(r1))
    return True
    

#%% A 6-dimensional triplet

A1 = np.array([
    [1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [1, 1, 0, 5, 0, 0],
    [2, 0, 1, 0, 5, 0],
    [1, 2, 1, 0, 0, 5],
])

A2 = np.array([
    [1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [2, 1, 0, 5, 0, 0],
    [0, 1, 1, 0, 5, 0],
    [3, 2, 1, 0, 0, 5],
])

A3 = np.array([
    [1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [2, 1, 0, 5, 0, 0],
    [0, 1, 1, 0, 5, 0],
    [2, 3, 1, 0, 0, 5],
])

Q1 = np.matmul(np.transpose(A1), A1)
Q2 = np.matmul(np.transpose(A2), A2)
Q3 = np.matmul(np.transpose(A3), A3)

print(isospectral_quadratic_forms(2*Q1, 2*Q2)) # Multiply by 2 to make them even
#print(isospectral_quadratic_forms(2*Q1, 2*Q3))
