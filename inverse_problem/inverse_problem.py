# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 19:50:20 2020

@author: Eduardo Thomaz
"""

import numpy as np
import matplotlib.pyplot as plt
import random

def le_arquivo(N):
    
    f = open("dados.txt", "r")
    
    p_str = f.readline().split()
    p = np.asarray(p_str,dtype=np.float32).reshape((len(p_str),1))

    conteudo = f.readlines()
    conteudo = [x.strip() for x in conteudo]
    ut = np.asarray(conteudo, dtype=np.float32).reshape(len(conteudo),1)
    
    dx = int((len(conteudo)-1)/N)
    ut_selecionado = []
    for i in range(0, len(conteudo), dx):
        ut_selecionado.append(ut[i,0])

    uT = np.asarray(ut_selecionado[1:-1], dtype=np.float32).reshape(-1, 1)
    
    f.close()
    return p, uT

#decompoe matriz simetrica A, em LDL^t
def ldlt2(A):
    n = len(A)
    v = np.zeros([1, n])
    
    d = np.zeros([1, n])
    L = np.ones([n, n])
    
    soma_vl_j = 0
    soma_vl_k = 0
    
    for i in range(0, n):
        for j1 in range(0, i):
            v[0, j1] = L[i, j1]*d[0, j1]
            soma_vl_j = soma_vl_j + L[i, j1]*v[0, j1]
             
        d[0, i] = A[i, i] - soma_vl_j
        soma_vl_j = 0
        
        for j2 in range(i+1, n):
            for k in range(0, i):
                soma_vl_k = soma_vl_k + L[j2, k]*v[0, k]
            
            L[j2, i] = (A[j2, i] - soma_vl_k)/d[0, i]
            soma_vl_k = 0
    
    L[L==1] = 0
    np.fill_diagonal(L, 1)
    
    D = np.diag(d[0])
    
    return L, D

#decompoe matriz tridiagonal A em LDLt            
def ldlt(A, a):
    #A e a np.array (1, ?/?-1)
    d = np.zeros((1, A.size))
    l = np.zeros((1, a.size))
                 
    d[0,0] = A[0,0]
    
    for i in range(0, l.size):
        l[0,i] = a[0,i]/d[0,i]
        d[0,i+1] = A[0,i+1] - d[0,i]*((l[0,i])**2) 
                     
    return l, d

#resolve o sistema A*x = b, se A for simetrica
def resolve_ldlt2(A, b):
    b = b.reshape([1, -1])
    n = len(A)
    L, D = ldlt2(A)
    
    soma_y = 0
    soma_x = 0
    
    y = np.zeros([1, n])
    y[0, 0] = b[0,0]
    
    for i in range(1, n):
        for j in range(0, i):
            soma_y = soma_y + L[i, j]*y[0, j]
        
        y[0, i] = b[0, i] - soma_y
        soma_y = 0
        
    z=y/np.diag(D).reshape(1, -1)
    
    x = np.zeros([1, n])
    x[0, -1] = z[0, -1]
    
    for i in range(n-2, -1, -1):
        for j in range(i+1, n):
            soma_x = soma_x + L[j, i]*x[0, j]
            
        x[0, i] = z[0, i] - soma_x
        soma_x = 0
        
    return x.reshape(-1, 1) 

#resolve o sistema A*x = b, se A for tridiagonal
def resolve_ldlt(A, a, b):
    l, d =  ldlt(A, a)
    
    y = np.zeros((1, l.size+1))
    y[0,0] = b[0,0]
    
    for i in range(0, l.size):
        y[0,i+1] = b[0,i+1] - l[0,i]*y[0,i]
    
    
    z = y/d
    
    z_inv = np.flip(z)
    l_inv = np.flip(l)
    x_inv = np.zeros((1, z.size))
    x_inv[0, 0] = z_inv[0, 0]
    
    for i in range(0, l.size):
        x_inv[0][i+1] = z_inv[0,i+1] - l_inv[0, i]*x_inv[0, i]
                                                        
    x = np.flip(x_inv)


    return x   

#distribuiçao inicial -> retorna 0 sempre
def u0(N):                   
    return np.zeros((1, N+1))

#temperatura na ponta 0 -> retorna 0 sempre
def g1():                     
    return 0

#temperatura na ponta 1 -> retorna 0 sempre
def g2():
    return 0

#retorna o falor de f(t, x)
def f(t, x, p):
    #t eh float on np.array (1,1)
    #x eh np.array(1, ?
    r = 10*(1+np.cos(5*t))

    dx = x[0,1] - x[0,0]
    h = dx
    g = x.copy()
        
    g[g<(p-h/2)] = 0
    g[g>(p+h/2)] = 0
    g[np.logical_and((g>=(p-h/2)),(g<=(p+h/2)))] = 1/h
    
    
    f = g*r
                      
    return f

def resolve_crank(N, p):
    ## Inicializacao
    T = 1
    M = N
    
    dx = 1/N
    dt = T/M
    
    x = np.arange(0, 1+dx/2, dx).reshape(1,N+1).reshape(1, -1)
    
    # calculo de lambda
    lambida = dt/(dx**2)
    
    #valores iniciais
    u_k = np.array([u0(N)[0, 1:-1]]) #so o meio da barra
    u_k1 = u_k.copy()
    
    # U guarda os valores que serao plotados
    U = u0(N) 
    
    #Matriz A
    A = np.ones((1, N-1))*(1+lambida)
    a = np.ones((1, N-2))*(-lambida/2)

    
    # Matriz C
    # para completar o sistema A*u_k = C*u_k1 + (f + g1 + g2)
    C = np.zeros((N-1, N-1))
    for i in range(0, len(C)):
        C[i, i] = 1-lambida
    
    for i in range(0, len(C)-1):
        C[i, i+1] = lambida/2
        C[i+1, i] = lambida/2
    
    #iteracao resolve problema
    for k in range(0, M):
        f_k = f((k)*dt, x[0, 1:-1].reshape(1, -1), p)
        f_k1 = f((k+1)*dt, x[0, 1:-1].reshape(1, -1), p)
        
        g1_k1 = g1()
        g1_k = g1()
        
        g2_k1 = g2()
        g2_k = g2()
        
        b = C.dot(u_k.transpose()) + dt*((f_k+f_k1).transpose())/2
        b[0,0]  = b[0,0]  + lambida*(g1_k1+g1_k)/2
        b[-1,0] = b[-1,0] + lambida*(g2_k1+g2_k)/2
        
        ###
        ###        
        u_k1 = resolve_ldlt(A, a, b.transpose())
        
        # salvando valores para plotar
        comeco = np.array([[g1()]]) #temp no comeco da barra
        fim    = np.array([[g2()]]) #temp no fim da barra
        U = np.append(U, np.append(comeco,np.append(u_k1, fim, axis=1), axis=1), axis=0)
    
        #resetando a iteracao
        u_k = u_k1.copy()
        
    return U[-1, 1:-1].reshape(1, -1)

#calculas cada u_k devida a cada fonte f_k
def gera_Uk(N, p):
    p = np.array([p]).reshape(1, -1)
    nf = np.size(p, 1)
    Uk = np.zeros([N-1, nf])
    
    for i in range(0, nf):
        Uk[:, i] = resolve_crank(N, p[0, i])
        
    return Uk
 
#calcula o sistema normal associado a Ax=b       
def cria_normal(A, b):
    A_normal = A.transpose().dot(A)
    b_normal = A.transpose().dot(b)

    return A_normal, b_normal
    
def erro_quadratico(N, uT, a, Uk):
    
    a = np.array([a]).reshape(-1, 1)
    dx=1/N
    
    u = np.dot(Uk, a)
    
    soma = 0
    for i in range(1, N-1):
        soma = soma + (uT[i, 0]-u[i, 0])**2
            
    return np.sqrt(dx*soma)   

def imprime(N, a, erro):
    a_str = ["a0","a1","a2","a3","a4","a5","a6","a7","a8","a9"]
    print("\nPara N = ", N, "as intensidades são:")
    for i in range(0, np.shape(a)[0]):
        print(a_str[i], " = ", a[i, 0])
    
    print("\nerro quadrático =", erro, "\n")

def printa(N, uT, a, Uk):
    a = np.array([a]).reshape(-1, 1)
    u = np.dot(Uk, a)
    
    dx=1/N
    
    x = np.arange(0, 1+dx/2, dx).reshape(-1, 1)
    u = np.append(np.append([0], u), [0]).reshape(-1, 1)
    uT = np.append(np.append([0], uT), [0]).reshape(-1, 1)
    
    fig, axs = plt.subplots(2)
    
    fig.suptitle("Temperatura vs. Posição")
    axs[0].plot(x, u,'m', linewidth=3, label='$\sin (x)$')
    axs[0].set_xlim((0,1))
    axs[0].set_ylim(bottom=0)
    axs[0].set_ylabel("Temperatura")
    
    axs[1].plot(x, uT, 'k', linewidth=0.5)
    axs[1].set_xlim((0,1))
    axs[1].set_ylim(bottom=0)
    axs[1].set_xlabel("Posição")
    axs[1].set_ylabel("Temperatura")
    
    fig.legend(['$u_{calculado}$', '$u_T$'])


    
teste = input("Qual fase(teste) você gostaria de jogar?\n")

if teste == "a":
    N = 128
    p = 0.35
    
    Uk = gera_Uk(N, p)
    uT = 7*Uk

    A_normal, b_normal = cria_normal(Uk, uT)
    a = resolve_ldlt2(A_normal, b_normal)
    
    erro = erro_quadratico(N, uT, a, Uk)
    imprime(N, a, erro)
    printa(N, uT, a, Uk)
    
elif teste == "b":
    N = 128
    p = [0.15, 0.3, 0.7, 0.8]
    
    Uk = gera_Uk(N, p)
    uT = 2.3*Uk[:, 0] + 3.7*Uk[:, 1] + 0.3*Uk[:, 2] + 4.2*Uk[:, 3]
    uT = uT.reshape(-1, 1)
    
    A_normal, b_normal = cria_normal(Uk, uT)
    a = resolve_ldlt2(A_normal, b_normal)
    
    erro = erro_quadratico(N, uT, a, Uk)
    imprime(N, a, erro)
    printa(N, uT, a, Uk)
    
elif teste == "c":
    N = int(input("Qual a dificuldade(N)?\n"))
    
    p, uT = le_arquivo(N)
    Uk = gera_Uk(N, p)
    
    A_normal, b_normal = cria_normal(Uk, uT)
    a = resolve_ldlt2(A_normal, b_normal)
    
    erro = erro_quadratico(N, uT, a, Uk)
    imprime(N, a, erro)
    printa(N, uT, a, Uk)
    
elif teste == "d":
    N = int(input("Qual a dificuldade(N)?\n"))
    e = 0.01
    
    p, uT = le_arquivo(N)
    
    for x in range(0, len(uT)):
        r = 2*(random.random()-0.5)
        uT[x, 0] = (1+e*r)*uT[x, 0]
        
    uT = uT.reshape(-1, 1)
    Uk = gera_Uk(N, p)
    
    A_normal, b_normal = cria_normal(Uk, uT)
    a = resolve_ldlt2(A_normal, b_normal)
    
    erro = erro_quadratico(N, uT, a, Uk)
    imprime(N, a, erro)
    printa(N, uT, a, Uk)             
                 
    
    
    
    
    

    
    
    
    
    
    
