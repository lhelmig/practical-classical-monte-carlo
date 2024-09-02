import numpy as np
import math

from numba import jit

# constants

h = 0
J = 1

# burn-in time
t0 = 1000
# total time interval in which measurement takes place
t_sample = 25000
# time steps between measuring the configuration
dt = 10

t_c = 2.269

# time is measured in units of monte-carlo sweeps, so t = 1 equals N attempted spin-flips

# creates an initial state where q denotes the share of down spins. So q = 1, creates a configuration, where all spins
# point downwards

@jit(nopython=True)
def create_state(q,L):
    
    state = np.ones(L*L)
    
    for i in range(0,math.floor(q*L*L)):
        
        state[i]=-1
        
    np.random.shuffle(state)
    
    return state

# the system is represented as a 1d array, so we need a 
# function that calculates the correct neighbour for the periodic lattice
@jit(nopython=True)
def get_left_neighbour(i,L):
    
    if(i%L==0):
        return i+L-1
    return i-1

@jit(nopython=True)
def get_right_neighbour(i,L):
    
    if((i+1)%L==0):
        return i-L+1
    return i+1

@jit(nopython=True)
def get_upper_neighbour(i,L):
    
    if(i>=L*L-L):
        return i%L
    return i+L

@jit(nopython=True)
def get_lower_neighbour(i,L):
    
    if(i<L):
        return i+L*(L-1)
    return i-L

# wrapper function, that returns the spins of all neighbours in an array

@jit(nopython=True)
def get_neighbours(i, state,L):
    return np.array([state[get_left_neighbour(i,L)],state[get_upper_neighbour(i,L)],state[get_right_neighbour(i,L)],state[get_lower_neighbour(i,L)]])

#calculates the magnetisation

@jit(nopython=True)
def magnetisation(state):
    
    return np.sum(state)

#calculates the energy

@jit(nopython=True)
def energy(state,L):
    
    U = 0
    
    U_interaction = 0

    
    # visit each site i, get the neighbours and calculate the interaction 
    
    for x in range(0,L*L):

        spin_i = state[x]
        
        U += h*spin_i
        
        neighbours = get_neighbours(x,state,L)

        U_interaction += np.sum(J*spin_i*neighbours)
        
    # each interaction is counted twice, so we need to half the interaction part of the energy

    U += 0.5 * U_interaction
    
    #return the negative energy
    
    return -U

#calculate the energy change

@jit(nopython=True)
def energy_change(state,i,L):
    
    # first, get the neighbours
    
    neighbours = get_neighbours(i,state,L)
    
    # dflip is (s'_i-s_i), which is either -2 if s_i was 1 and 2 if s_i was -1
    rd_spin = state[i]
    dflip = (-2)*rd_spin
    
    # energy change, that results from a spin flip
    
    # we need only to sum up the spins of the neighbours and multiply with dflip
    
    dH = -h*dflip - J *dflip*(np.sum(neighbours))
    
    return dH,rd_spin

# implements one entire monte carlo sweep

@jit(nopython=True)   
def metropolis_procedure(state,curr_t,L):
    
    # one monte-carlo sweep

    for i in range(L*L):
        
        # pick random site
        
        rd_site = np.random.randint(0,L*L)

        # calculate energy change
        
        dH, rd_spin = energy_change(state,rd_site,L)

        # if energy change is negative, flip spin
        # if not, draw a random number, and then maybe flip the spin
        
        if(dH>0):
            
            #draw random number
            p = np.random.random_sample()
            # if p < e^-dH/T, then flip the spin
            if(p<np.exp(-dH/curr_t)):
                #flip the spin
                state[rd_site]=-1*rd_spin
        else:
            state[rd_site]=-1*rd_spin

# implements the metropolis algorithm for an input temperature T
# after burn-in time t_0, sampling starts
# 250.000 monte carlo sweeps are done
# after 10 monte carlo sweeps: measurement of m,e,m^2 and e^2         

@jit(nopython=True)

def metropolis(T,state,L):
    
    n_samples = 0
    
    U = 0
    U_sq = 0
    
    M = 0
    M_sq = 0
    
    # burn in phase
    
    for t in range(0,t0):
        
        metropolis_procedure(state,T,L)
    
    print("burn-in phase finished")
    
    # sampling starts
    
    for t in range(t0,t_sample):
        
        # do one monte carlo sweep
        
        metropolis_procedure(state,T,L)
        
        # measure after dt monte carlo sweeps
        
        if(t%dt==0):
            
            n_samples = n_samples + 1
            
            m = magnetisation(state)
            
            M += m
            
            M_sq += m*m
            
            e = energy(state,L)
            
            U += e
            U_sq += e*e
            
    # calc the average of the sums
    
    U = U/n_samples
    M = M/n_samples
    U_sq = U_sq/n_samples
    M_sq = M_sq/n_samples
    
    print("T = ",T," finished")
    
    return [L,T,U,U_sq,M,M_sq,0,np.abs(M),(T-t_c)/t_c,(U_sq-U**2)/(T**2),(M_sq-M*M)/T,(M_sq-M*M)/T]

temp_interval = np.linspace(2.0,2.5,100)

L = 20

state = create_state(1,L)

data = [metropolis(T,state,L) for T in temp_interval]

L = 30

state = create_state(1,L)

data += [metropolis(T,state,L) for T in temp_interval]

L = 40
N = L**2

state = create_state(1,L)

data += [metropolis(T,state,L) for T in temp_interval]

L = 50
N = L**2

state = create_state(1,L)

data += [metropolis(T,state,L) for T in temp_interval]

L = 60

state = create_state(1,L)

data += [metropolis(T,state,L) for T in temp_interval]

df = pd.DataFrame(data,columns=['L','T','e','e2','m','m2','m4','absem','t','c','chi','chiabs'])

df['lattice']='square'

df.to_csv("data_square.csv", index = False)