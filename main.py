import numpy as np
import matplotlib.pyplot as plt
import time
import sys

N = 10
K = 10
S = 10

neurons            = xrange(0, N)
neuron_coupling    = np.random.normal(loc=K, scale=S,       size=(N, N))
neuron_frequencies = np.random.uniform(low=0, high=1,       size=(N))
neuron_phase       = np.random.uniform(low=0, high=2*np.pi, size=(N))

neuron_frequencies2 = np.ones(shape=(N))

neuron_coupling    = np.abs(neuron_coupling)
for x in neurons:
  neuron_coupling[x,x] = 0

def kuramoto(n):
    global neuron_frequencies
    global neuron_phase
    global N
    # start with our 'natural' frequency
    phase_change = neuron_frequencies[n]

    # all the other neurons contribute
    for x in neurons:
        phase_change = phase_change + (neuron_coupling[n,x] \
                     * np.sin(neuron_phase[n] - neuron_phase[x]))/N
    
    # normalize
    return  phase_change

t = np.array(0)
n = [np.array(0)] * N

colors = [np.random.rand(3,1)] * N
for c in neurons:
    colors[c] = np.random.rand(3,1)
# simulation

plt.ion()
plt.show()

MAX  = 3000
STEP = 0.01
SKIP = 100
for i in xrange(0, 100000000):
    t    = np.append(t, STEP*i)
    if t.size > MAX: t = np.delete(t, 0)
    for x in neurons:
        neuron_frequencies2[x] = kuramoto(x)
        neuron_phase[x]        = neuron_phase[x] + neuron_frequencies2[x] * STEP
        neuron_phase[x]        = neuron_phase[x] % (2 * np.pi)
        n[x] = np.append(n[x], [neuron_phase[x]])
        if n[x].size > MAX: n[x] = np.delete(n[x], 0)

    if i % SKIP == 0:
        plt.figure(1)
        plt.clf()
        plt.subplot(221)
        for j in neurons:
            plt.plot(t, np.sin(n[j]), c=colors[j])

        plt.subplot(222)
        plt.imshow(neuron_coupling, interpolation='nearest', cmap=plt.cm.jet)
        plt.colorbar()

        plt.subplot(223)
        plt.imshow(np.vstack((neuron_frequencies,neuron_frequencies2)), 
                   interpolation='nearest', cmap=plt.cm.jet)
        plt.colorbar()

        amplitude = [] 
        phase     = []
        for j in neurons:
            amplitude = np.append(amplitude, np.cos(neuron_phase[j]))
            phase     = np.append(phase, np.sin(neuron_phase[j]))
        plt.subplot(224)
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.scatter(amplitude, phase, color='b')
        plt.scatter(amplitude.mean(), phase.mean(), color='r')
      
        plt.draw()
