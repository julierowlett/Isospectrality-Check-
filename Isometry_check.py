import numpy as np
import itertools
import time
from itertools import permutations, product

def generate_vectors(norm_limit, dimensions):
    vectors = []
    norm_limit_int = int(np.ceil(norm_limit))
    for values in itertools.product(range(-norm_limit_int, norm_limit_int + 1), repeat = dimensions):
        vector = np.array(values)
        norm = np.linalg.norm(vector)
        if norm <= norm_limit:
            vectors.append(vector)
    return vectors

def create_matrix_from_vectors(*vectors):
    vectors = np.array(vectors)

    if len(vectors[0].shape) == 1:
        matrix = np.column_stack(vectors)
    else:
        matrix = np.vstack(vectors)

    return matrix

def congruent_lattices(A1, A2):
    n = len(A1)
    m = len(A2)
    if n != m:
        raise Exception("Dimensions must be equal")
    Q1 = np.matmul(np.transpose(A1), A1)
    Q2 = np.matmul(np.transpose(A2), A2)
    return integrally_equivalent_6d_quadratic_forms(Q1, Q2)

def integrally_equivalent_6d_quadratic_forms(Q1, Q2): # Only works in 6d
    n = len(Q1)
    m = len(Q2)
    if n != m:
        raise Exception("Dimensions must be equal")
    eigenvalues1, _ = np.linalg.eig(Q1)
    eigenvalues2, _ = np.linalg.eig(Q2)
    eigenvalues1 = np.real(eigenvalues1)
    eigenvalues2 = np.real(eigenvalues2)
    lambda_min1 = np.min(eigenvalues1)
    lambda_min2 = np.min(eigenvalues2)
    
    if lambda_min1 <= 0 or lambda_min2 <= 0:
        raise Exception("Q1 and Q2 are not positive definite")
    
    norm_limits = []
    for i in range(0,n):
        norm_limits.append(np.sqrt(Q2[i,i]/lambda_min1))
    
    
    ##################### Congruency in 6d
    if n == 6:
        vectors0 = generate_vectors(norm_limits[0], n)
        vectors1 = generate_vectors(norm_limits[1], n)
        vectors2 = generate_vectors(norm_limits[2], n)
        vectors3 = generate_vectors(norm_limits[3], n)
        vectors4 = generate_vectors(norm_limits[4], n)
        vectors5 = generate_vectors(norm_limits[5], n)
        
        N0 = len(vectors0)
        N1 = len(vectors1)
        N2 = len(vectors2)
        N3 = len(vectors3)
        N4 = len(vectors4)
        N5 = len(vectors5)
        N = N0*N1*N2*N3*N4*N5
        
        print("Now we have " + str(N0) + "*" + str(N1) + "*" + str(N2) + "*" + str(N3) + "*" + str(N4) + "*" + str(N5) + " = " + str(N) + " columns")
        
        vectors0_n = []
        vectors1_n = []
        vectors2_n = []
        vectors3_n = []
        vectors4_n = []  
        vectors5_n = [] 
        
        for v0 in vectors0:
            if np.matmul(np.transpose(v0), np.matmul(Q1, v0)) == Q2[0,0]:
               vectors0_n.append(v0)
               
        for v1 in vectors1:
            if np.matmul(np.transpose(v1), np.matmul(Q1, v1)) == Q2[1,1]:
               vectors1_n.append(v1)
               
        for v2 in vectors2:
            if np.matmul(np.transpose(v2), np.matmul(Q1, v2)) == Q2[2,2]:
               vectors2_n.append(v2)
               
        for v3 in vectors3:
            if np.matmul(np.transpose(v3), np.matmul(Q1, v3)) == Q2[3,3]:
               vectors3_n.append(v3)
        
        for v4 in vectors4:
            if np.matmul(np.transpose(v4), np.matmul(Q1, v4)) == Q2[4,4]:
               vectors4_n.append(v4)
               
        for v5 in vectors5:
            if np.matmul(np.transpose(v5), np.matmul(Q1, v5)) == Q2[5,5]:
               vectors5_n.append(v5)
        
        # Remove matrices B which don't satisfy b_0^T*Q1*b_j = (Q_2)_{0j} 
        vectors0_nn = []
        vectors1_nn = []
        vectors2_nn = []
        vectors3_nn = []
        vectors4_nn = [] 
        vectors5_nn = [] 
        
        for v0 in vectors0_n:
            for v1 in vectors1_n:
                if np.matmul(np.transpose(v0), np.matmul(Q1, v1)) == Q2[0,1]:
                   if any(np.array_equal(v0, mat) for mat in vectors0_nn) == False:
                       vectors0_nn.append(v0)
                   if any(np.array_equal(v1, mat) for mat in vectors1_nn) == False:
                       vectors1_nn.append(v1)
                   
        for v0 in vectors0_nn:
            for v2 in vectors2_n:
                if np.matmul(np.transpose(v0), np.matmul(Q1, v2)) == Q2[0,2]:
                   if any(np.array_equal(v2, mat) for mat in vectors2_nn) == False:
                       vectors2_nn.append(v2)
                   
        for v0 in vectors0_nn:
            for v3 in vectors3_n:
                if np.matmul(np.transpose(v0), np.matmul(Q1, v3)) == Q2[0,3]:
                   if any(np.array_equal(v3, mat) for mat in vectors3_nn) == False:
                       vectors3_nn.append(v3)
                   
        for v0 in vectors0_nn:
            for v4 in vectors4_n:
                if np.matmul(np.transpose(v0), np.matmul(Q1, v4)) == Q2[0,4]:
                   if any(np.array_equal(v4, mat) for mat in vectors4_nn) == False:
                       vectors4_nn.append(v4)
                       
        for v0 in vectors0_nn:
            for v5 in vectors5_n:
                if np.matmul(np.transpose(v0), np.matmul(Q1, v5)) == Q2[0,5]:
                   if any(np.array_equal(v5, mat) for mat in vectors5_nn) == False:
                       vectors5_nn.append(v5)
             
        # Remove matrices B which don't satisfy b_1^T*Q1*b_j = (Q_2)_{1j} 
        vectors0_nnn = vectors0_nn
        vectors1_nnn = []
        vectors2_nnn = []
        vectors3_nnn = []
        vectors4_nnn = [] 
        vectors5_nnn = [] 
        
        for v1 in vectors1_nn:
            for v2 in vectors2_nn:
                if np.matmul(np.transpose(v1), np.matmul(Q1, v2)) == Q2[1,2]:
                   if any(np.array_equal(v1, mat) for mat in vectors1_nnn) == False:
                       vectors1_nnn.append(v1)
                   if any(np.array_equal(v2, mat) for mat in vectors2_nnn) == False:
                       vectors2_nnn.append(v2)
                   
        for v1 in vectors1_nnn:
            for v3 in vectors3_nn:
                if np.matmul(np.transpose(v1), np.matmul(Q1, v3)) == Q2[1,3]:
                   if any(np.array_equal(v3, mat) for mat in vectors3_nnn) == False:
                       vectors3_nnn.append(v3)
                   
        for v1 in vectors1_nnn:
            for v4 in vectors4_nn:
                if np.matmul(np.transpose(v1), np.matmul(Q1, v4)) == Q2[1,4]:
                   if any(np.array_equal(v4, mat) for mat in vectors4_nnn) == False:
                       vectors4_nnn.append(v4)
                       
        for v1 in vectors1_nnn:
            for v5 in vectors5_nn:
                if np.matmul(np.transpose(v1), np.matmul(Q1, v5)) == Q2[1,5]:
                   if any(np.array_equal(v5, mat) for mat in vectors5_nnn) == False:
                       vectors5_nnn.append(v5)
        
        # Remove matrices B which don't satisfy b_2^T*Q1*b_j = (Q_2)_{2j} 
        vectors0_nnnn = vectors0_nnn
        vectors1_nnnn = vectors1_nnn
        vectors2_nnnn = []
        vectors3_nnnn = []
        vectors4_nnnn = [] 
        vectors5_nnnn = [] 
        
        for v2 in vectors2_nnn:
            for v3 in vectors3_nnn:
                if np.matmul(np.transpose(v2), np.matmul(Q1, v3)) == Q2[2,3]:
                   if any(np.array_equal(v2, mat) for mat in vectors2_nnnn) == False:
                       vectors2_nnnn.append(v2)
                   if any(np.array_equal(v3, mat) for mat in vectors3_nnnn) == False:
                       vectors3_nnnn.append(v3)
                       
        for v2 in vectors2_nnnn:
            for v4 in vectors4_nnn:
                if np.matmul(np.transpose(v2), np.matmul(Q1, v4)) == Q2[2,4]:
                   if any(np.array_equal(v4, mat) for mat in vectors4_nnnn) == False:
                       vectors4_nnnn.append(v4)
                       
        for v2 in vectors2_nnnn:
            for v5 in vectors5_nnn:
                if np.matmul(np.transpose(v2), np.matmul(Q1, v5)) == Q2[2,5]:
                   if any(np.array_equal(v5, mat) for mat in vectors5_nnnn) == False:
                       vectors5_nnnn.append(v5)
        
        # Remove matrices B which don't satisfy b_3^T*Q1*b_j = (Q_2)_{3j} 
        vectors0_nnnnn = vectors0_nnnn
        vectors1_nnnnn = vectors1_nnnn
        vectors2_nnnnn = vectors2_nnnn
        vectors3_nnnnn = []
        vectors4_nnnnn = [] 
        vectors5_nnnnn = [] 
        
        for v3 in vectors3_nnnn:
            for v4 in vectors4_nnnn:
                if np.matmul(np.transpose(v3), np.matmul(Q1, v4)) == Q2[3,4]:
                   if any(np.array_equal(v3, mat) for mat in vectors3_nnnnn) == False:
                       vectors3_nnnnn.append(v3)
                   if any(np.array_equal(v4, mat) for mat in vectors4_nnnnn) == False:
                       vectors4_nnnnn.append(v4)
                       
        for v3 in vectors2_nnnnn:
            for v5 in vectors5_nnnn:
                if np.matmul(np.transpose(v3), np.matmul(Q1, v5)) == Q2[3,5]:
                   if any(np.array_equal(v5, mat) for mat in vectors5_nnnnn) == False:
                       vectors5_nnnnn.append(v5)
        
        # Remove matrices B which don't satisfy b_4^T*Q1*b_j = (Q_2)_{4j} 
        vectors0_nnnnnn = vectors0_nnnnn
        vectors1_nnnnnn = vectors1_nnnnn
        vectors2_nnnnnn = vectors2_nnnnn
        vectors3_nnnnnn = vectors2_nnnnn
        vectors4_nnnnnn = [] 
        vectors5_nnnnnn = [] 
        
        for v4 in vectors4_nnnnn:
            for v5 in vectors5_nnnnn:
                if np.matmul(np.transpose(v4), np.matmul(Q1, v5)) == Q2[4,5]:
                   if any(np.array_equal(v4, mat) for mat in vectors4_nnnnnn) == False:
                       vectors4_nnnnnn.append(v4)
                   if any(np.array_equal(v5, mat) for mat in vectors5_nnnnnn) == False:
                       vectors5_nnnnnn.append(v5)
        
        
        N0 = len(vectors0_nnnnnn)
        N1 = len(vectors1_nnnnnn)
        N2 = len(vectors2_nnnnnn)
        N3 = len(vectors3_nnnnnn)
        N4 = len(vectors4_nnnnnn)
        N5 = len(vectors5_nnnnnn)
        N = N0*N1*N2*N3*N4*N5
        print("Now we only have " + str(N0) + "*" + str(N1) + "*" + str(N2) + "*" + str(N3) + "*" + str(N4) + "*" + str(N5) + " = " + str(N) + " matrices to check") 
        for v0 in vectors0_nnnnnn:
            for v1 in vectors1_nnnnnn:
                for v2 in vectors2_nnnnnn:
                    for v3 in vectors3_nnnnnn:
                        for v4 in vectors4_nnnnnn:
                            for v5 in vectors5_nnnnnn:
                                B = create_matrix_from_vectors(v0, v1, v2, v3, v4, v5)
                                detB = np.linalg.det(B)
                                if np.array_equal( np.matmul(np.transpose(B), np.matmul(Q1, B)), Q2):
                                    print("Matrix " + str(B) + " works")
                                    return True
        return False
    

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

print(congruent_lattices(A1, A2))
#print(congruent_lattices(A1, A3))
#print(congruent_lattices(A2, A3))

