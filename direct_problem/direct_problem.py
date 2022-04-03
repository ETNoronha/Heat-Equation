# -*- coding: utf-8 -*-
"""
Created on Thu May 14 16:50:57 2020

@author: Eduardo Thomaz
"""
import numpy as np
import matplotlib.pyplot as plt

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


    
#retorna o falor de f, dependendo do teste.
#t deve ser float
#x pode ser float(metodo explicito) ou array(outros)
def f(t, x, teste, dx=0.1):
    #t eh float on np.array (1,1)
    #x eh np.array(1, ?
    
    x = np.array([x])
    x = np.array([x.flatten()])
    if teste == 'a':
        #f = 10*(x**2)*(x-1) -60*x*t +20*t
        f = 10*np.cos(10*t)*x**2*(1-x)**2 - ((1+np.sin(10*t))*(12*x**2 - 12*x + 2))
        
    elif teste == 'b':
        f = np.exp(t-x)*(25*(t**2)*np.cos(5*t*x) - 10*t*np.sin(5*t*x) - 5*x*np.sin(5*t*x))
        
    elif teste == 'c':
        r = 10*(1+np.cos(5*t))
        
        p = 0.2
        dx = x[0,1] - x[0,0]
        h = dx
        g = x.copy()
            
        g[g<(p-h/2)] = 0
        g[g>(p+h/2)] = 0
        g[np.logical_and((g>=(p-h/2)),(g<=(p+h/2)))] = 1/h
    
    
        f = g*r
                      
    return f


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

#retorna o valor da temperatura ao longo da barra em t=0
def u0(teste, x, N):
    #
    #x eh np.array(1, N+1)
    
    #questao a
    if teste == 'a' :
        u0 = (x**2)*((1-x)**2)
        
    elif teste == 'b':
        u0 = np.exp(-x)
        
    elif teste == 'c':
        u0 = np.zeros((1, N+1))
                      
    return u0
    
    
#funcao g1
def g1(t, teste):
    #t eh float
    if teste == 'a':
        g1 = 0;
        
        
    elif teste == 'b':
        g1 = np.exp(t)
        
    elif teste == 'c':
        g1 = 0;
                      
    return g1
    
#funcao g2    
def g2(t, teste):
    #t eh float
    if teste == 'a':
        g2 = 0;
             
    elif teste == 'b':
        g2 = np.exp(t-1)*np.cos(5*t)
        
    elif teste == 'c':
        g2 = 0;
                      
    return g2
    
    
#retorna o erro da aproximacao calculada
def calcula_erro(teste, u_final, x):
    if teste == 'a':
        #u_exato em t = T = 1
        u_exato = (1 + np.sin(10))*(x**2)*((1-x)**2)
             
    elif teste == 'b':
        u_exato = np.exp(1-x)*np.cos(5*x)
        
    elif teste == 'c':
       return "Fí, Fá, Fó, Folso. Este erro não está em nosso arcabouço!"
       
                      
    #return abs((u_exato - u_final).max())
    return (abs(u_exato - u_final)).max()


def plota_heatmap(U):
    plt.title("Mapa de Calor")
    plt.xlabel("Posição")
    plt.ylabel("Tempo")
            
    plt.imshow(U, aspect='auto', origin='lower', extent=[0,1,0,1])
        
    cores = plt.colorbar()
    cores.set_label("Temperatura")
                    
    plt.show()
    
    

#resolve a aproximacao para o metodo explicito  
def resolve_explicito(teste, N, lambida):
    ## Inicializacao
    T = 1
    M = int(np.ceil((N**2 * T)/lambida))
    
    dx = 1/N
    dt = T/M
    
    x = np.arange(0, 1+dx, dx).reshape(1,N+1)
    
    # recalculo de lambda para que M seja inteiro
    lambida = dt/(dx**2)
    
    u_k = u0(teste, x, N)
    u_k1 = u_k.copy()
     
    # U guarda os valores que serao plotados
    U = u_k.copy()
    ## quantos pontos se quer plotar -> valor aproximado ###
    pontos = 100
    dm_1 = np.floor((M-1)/pontos) # queremos pegar no minimo 100 tempos
    
    for k in range(0, M):
        for i in range(1, N):
            u_k1[0, i] = u_k[0 ,i] + dt*(((u_k[0, i-1] - 2*u_k[0, i] + u_k[0, i+1])/dx**2) + f(k*dt, i*dx, teste, dx))
            
        #condicoes de contorno
        u_k1[0, 0]  = g1(k*dt, teste)
        u_k1[0, -1] = g2(k*dt, teste)
        
        #resetando a interacao
        u_k = u_k1.copy()
        
        # salvando valores para plotar
        if k%dm_1 == 0:
            U = np.append(U, u_k1, axis = 0)
            
    
    U = np.append(U, u_k1, axis = 0)
    plota_heatmap(U)
    
    return calcula_erro(teste, u_k1, x), M, lambida #erro    

#resolve a aproximacao para o metodo de euler    
def resolve_euler(teste, N):
    ## Inicializacao
    T = 1
    M = N
    
    dx = 1/N
    dt = T/M
    
    x = np.arange(0, 1+dx/2, dx).reshape(1,N+1)
    
    # calculo de lambda para que M seja inteiro
    lambida = dt/(dx**2)
    
    #valores iniciais
    u_k = np.array([u0(teste, x, N)[0, 1:-1]]) #so o meio da barra
    u_k1 = u_k.copy()
    
    # U guarda os valores que serao plotados
    U = u0(teste, x, N) 
    
    #Matriz A
    A = np.ones((1, N-1))*(1+2*lambida)
    a = np.ones((1, N-2))*(-lambida)
    
    
    for k in range(0, M):
        b = u_k + dt*f((k+1)*dt, x[0, 1:-1], teste, dx)
        b[0,0]  = b[0,0]  + lambida*g1((k+1)*dt, teste)
        b[0,-1] = b[0,-1] + lambida*g2((k+1)*dt, teste)
        
        u_k1 = resolve_ldlt(A, a, b)
        
        #resetando a iteracao
        u_k = u_k1.copy()
        
        # salvando valores para plotar
        
        comeco = np.array([[g1((k+1)*dt, teste)]]) #temp no comeco da barra
        fim    = np.array([[g2((k+1)*dt, teste)]]) #temp no fim da barra
        U = np.append(U, np.append(comeco,np.append(u_k1, fim, axis=1), axis=1), axis=0)
    
    
    plota_heatmap(U)
    
    return calcula_erro(teste, U[-1, :], x), M, lambida #erro 
    
#resolve a aproximacao para o metodo crank-nicolson
def resolve_crank(teste, N):
    ## Inicializacao
    T = 1
    M = N
    
    dx = 1/N
    dt = T/M
    
    x = np.arange(0, 1+dx/2, dx).reshape(1,N+1)
    
    # calculo de lambda para que M seja inteiro
    lambida = dt/(dx**2)
    
    #valores iniciais
    u_k = np.array([u0(teste, x, N)[0, 1:-1]]) #so o meio da barra
    u_k1 = u_k.copy()
    
    # U guarda os valores que serao plotados
    U = u0(teste, x, N) 
    
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
        f_k = f((k)*dt, x[0, 1:-1], teste, dx)
        f_k1 = f((k+1)*dt, x[0, 1:-1], teste, dx)
        
        g1_k1 = g1((k+1)*dt, teste)
        g1_k = g1((k)*dt, teste)
        
        g2_k1 = g2((k+1)*dt, teste)
        g2_k = g2((k)*dt, teste)
        
        b = C.dot(u_k.transpose()) + dt*((f_k+f_k1).transpose())/2
        b[0,0]  = b[0,0]  + lambida*(g1_k1+g1_k)/2
        b[-1,0] = b[-1,0] + lambida*(g2_k1+g2_k)/2
        
        u_k1 = resolve_ldlt(A, a, b.transpose())
        
        # salvando valores para plotar
        comeco = np.array([[g1((k+1)*dt, teste)]]) #temp no comeco da barra
        fim    = np.array([[g2((k+1)*dt, teste)]]) #temp no fim da barra
        U = np.append(U, np.append(comeco,np.append(u_k1, fim, axis=1), axis=1), axis=0)
    
        #resetando a iteracao
        u_k = u_k1.copy()
        
    
    plota_heatmap(U)
    
    return calcula_erro(teste, U[-1, :], x), M, lambida, U #erro 

def printar(N, M, lambida, teste, metodo, T, erro):
 
    espaco = ' '

    print("\n")
    print("+-----------------------+-----------------------+")
    print("| Método                |", metodo, (20-len(str(metodo)))*espaco,"|")
    print("+-----------------------+-----------------------+")
    print("| Teste                 |", teste, (20-len(str(teste)))*espaco,"|")
    print("+-----------------------+-----------------------+")
    print("| T                     |", T, (20-len(str(T)))*espaco,"|")
    print("+-----------------------+-----------------------+")
    print("| N                     |", N, (20-len(str(N)))*espaco,"|")
    print("+-----------------------+-----------------------+")
    print("| M                     |", M, (20-len(str(M)))*espaco,"|")
    print("+-----------------------+-----------------------+")
    print("| lambda                |", lambida, (20-len(str(lambida)))*espaco, "|")
    print("+-----------------------+-----------------------+")
    print("| Erro                  |", erro, (20-len(str(erro)))*espaco,"|")
    print("+-----------------------+-----------------------+")

    return
    
    
    
#define
Explicito = 1
Euler = 2
Crank_Nicolson = 3

#--------------Rotina Principal----------------#

#Inputs do usuario
N = int(input("Quantos feijões mágicos você por esta vaca?(valor de N) :"))

teste = input("Onde você vai plantar seus feijões:\na-No quintal(teste a)\nb-Num algodãozinho(teste b)\nc-No pasto(teste c)\n")
metodo = int(input("Qual desse você prefere:\n1-A galinha dos ovos dourados(Explícito)\n2-A harpa mágica(Euler)\n3-Nota 10 no EP(Crank-Nicolson)\nEscolha do método, digite o número: "))
T = 1

#Dependendo do metodo escolhido, chama a funcao que resolve o problema
if metodo == Explicito:
    lambida = float(input("De 0 a 10, quão bem escala pés de feijão?(valor de lambda) :"))
    erro, M, lambida = resolve_explicito(teste, N, lambida)
    metodo = "Explicito"
elif metodo == Euler:
    erro, M, lambida = resolve_euler(teste, N)
    metodo = "Euler"
elif metodo == Crank_Nicolson:
    erro, M, lambida, U = resolve_crank(teste, N)
    metodo = "Crank-Nicolson"
    
#printa a tabela com os resultados e parametros usados
printar(N, M, lambida, teste, metodo,  T, erro)


#----------End-----------#  
