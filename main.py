import numpy as np
import matplotlib.pyplot as plt
import time

N = 16

def a(t):
    return np.cos(0.1 * 2*np.pi*t)

def b(t):
    return np.cos(0.5 * 2*np.pi*t)

neurons            = xrange(0, N)
neuron_coupling    = np.random.normal(loc=100, scale=1,      size=(N, N))
neuron_frequencies = np.random.uniform(low=0, high=20,      size=(N))
neuron_phase       = np.random.uniform(low=0, high=2*np.pi, size=(N))

neuron_coupling    = np.tril(neuron_coupling)
for x in neurons:
  neuron_coupling[x,x] = 0

def kuramoto(n):
    # start with our 'natural' frequency
    phase_change = neuron_frequencies[n]

    # all the other neurons contribute
    for x in neurons:
        phase_change = phase_change + neuron_coupling[n,x] \
                     * np.sin(neuron_phase[n] - neuron_phase[x])
    
    # normalize
    return  phase_change/N

t = np.array(0)
n = [np.array(0)] * N

colors = [np.random.rand(3,1)] * N
for c in neurons:
    colors[c] = np.random.rand(3,1)
# simulation

plt.ion()
plt.show()

STEP = 0.1
for i in xrange(0, 10000):
    t    = np.append(t,    [i*STEP])
    if t.size > 20: t = np.delete(t, 0)
    for x in neurons:
        neuron_frequencies[x] = kuramoto(x)
        neuron_phase[x]       = neuron_phase[x] + neuron_frequencies[x] * STEP
        n[x] = np.append(n[x], [neuron_phase[x]])
        if n[x].size > 20: n[x] = np.delete(n[x], 0)
    
    plt.figure(1)
    plt.clf()
    plt.subplot(221)
    #plt.plot(t, a(t), 'b')
    #plt.plot(t, b(t), 'r')
    for j in neurons:
        f = np.sin(n[j])
        plt.plot(t, f, c=colors[j])

    plt.subplot(222)
    plt.imshow(neuron_coupling, interpolation='nearest', cmap=plt.cm.ocean)
    plt.colorbar()

    plt.subplot(223)
    plt.imshow(np.vstack((neuron_frequencies,neuron_frequencies)), 
               interpolation='nearest', cmap=plt.cm.ocean)
    plt.colorbar()
    plt.draw()
    time.sleep(0.02)
