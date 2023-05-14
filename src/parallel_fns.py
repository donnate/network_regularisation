from threading import local
import numpy as np
from multiprocessing import shared_memory, Process, Lock
import time


def print_func2(continent):
    #print('The name of continent is : ', continent)
    return continent

def print_func(continent='Asia'):
    #print('The name of continent is : ', continent)
    return continent

def test_fn(queue, value): 
    queue.put(value)

def compute_update(Q, u, b, processors, max_vals,  lambda1, index, queue):
    update = -(np.dot(Q[index], u) + b[index])/(processors*max_vals[index])
    #update after projection: 
    new_update = np.sign(u[index] + update) * min(u[index] + update, lambda1)
    queue.put((index, new_update))

def f(l, n, a, b, arg):
    n.value = 3.1415927
    while True:
        l.acquire()
        try:
            a[arg] +=1
            print(a[:])
            print(np.array(a[:]) - np.array(b[:]))
            if a[arg] > 5000: 
                break
        finally:
            print("lock released")
            l.release()
        

def add_one(shr_name, lock):
    existing_shm = shared_memory.SharedMemory(name=shr_name)
    np_array = np.ndarray((dim, dim,), dtype=np.int64, buffer=existing_shm.buf)
    lock.acquire()
    np_array[:] = np_array[0] + 1
    lock.release()
    time.sleep(10) # pause, to see the memory usage in top
    print('added one')
    existing_shm.close()

def project_op(vector, param): 
    vector[vector > param] = param
    vector[vector < -param] = -param
    return vector

def compute_and_update(u_array, grad_array, Q, update_vals, epsilon, lambda1, index1, index2): 
    print("has the fucking thing started yet")
    #Q = np.array(Qarr).reshape(Qarr.shape[1], Qarr.shape[1]) #local copy of Q
    width_Q = int(np.sqrt(len(Q)))
    print("wow it's fucking started") 
    local_u = np.array(u_array[:])
    local_grad = np.array(grad_array[:])

    while True:
        #print(u_array)
        projected_gradient = local_u - project_op(local_u - local_grad, lambda1)
        #print(projected_gradient)
        #print(index1, index2)

        greedy_coord = np.argmax(np.abs(projected_gradient[index1:index2])) #projected[grad:]\
        i = greedy_coord
        #print("greedy coord:" , greedy_coord)
        delta = min(max(local_u[i] - ((1/Q[(i*width_Q) + i]) * local_grad[i]), -lambda1), lambda1) - local_u[i]
        local_grad += delta*np.array(Q[i*width_Q: (i*width_Q) + width_Q])
        local_u += delta
        grad_array += delta*np.array(Q[i*width_Q: (i*width_Q) + width_Q]) #test lock speed here
        u_array[i] += delta
        update_vals.value +=1
        #print("current update number:", update_vals.value)
        print(delta)
        if abs(delta) <= epsilon:
            print("break reason 1")
            update_vals.value = 0
            break
