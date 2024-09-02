import numpy as np
import math
import random

from numba import jit

# constants

# burn-in time

t0 = 1000

# total time interval in which measurement takes place

# time is measured in units of monte-carlo sweeps, so t = 1 equals N attempted spin-flips

# creates an initial state where q denotes the share of down spins. So q = 1, creates a configuration, where all spins
# point downwards

@jit(nopython=True)
def create_state(q,L):
    
    state = np.ones(L*L)
    
    for i in range(0,math.floor(q*L*L)):
        
        state[i]=-1.0
        
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

##################

# to change the square lattice to a triangular lattice, we only need to add two new neighbours
# x - x - x - x - x - x
# | / | / | / | / | / |
# x - x - x - x - x - x
# | / | / | / | / | / |
# x - x - x - x - x - x

@jit(nopython=True)
def get_upper_right_neighbour(i,L):
    if(i == L*L-1): # right corner
        return 0
    if(i>=L*L-L):
        return (i%L)+1 # upper row, neighbour is in the first row one to the right
    if((i+1)%L==0): # right most row, neighbour is in the next row
        return i+1
    return i+L+1

@jit(nopython=True)

def get_lower_left_neighbour(i,L):
    if(i == 0): # left lower corner
        return L*L-1 # just return right upper corner
    if(i<L):
        return i+L*(L-1)-1
    if(i%L==0):
        return i-1
    return i-L-1

#########################

# wrapper function, that returns the spins of all neighbours in an array
@jit(nopython=True)
def get_neighbours(i, state,L):
    return np.array([state[get_left_neighbour(i,L)],state[get_upper_neighbour(i,L)],state[get_right_neighbour(i,L)],state[get_lower_neighbour(i,L)],state[get_upper_right_neighbour(i,L)],state[get_lower_left_neighbour(i,L)]])

#calculates the magnetisation

@jit(nopython=True)
def magnetisation(state):
    
    return np.sum(state)

#calculates the energy

@jit(nopython=True)
def energy(state,L,J,h):
    
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
def energy_change(state,i,L,J,h):
    
    # first, get the neighbours
    
    neighbours = get_neighbours(i,state,L)
    
    # dflip is (s'_i-s_i), which is either -2 if s_i was 1 and 2 if s_i was -1
    rd_spin = state[i]
    dflip = (-2)*rd_spin
    
    # energy change, that results from a spin flip
    
    # we need only to sum up the spins of the neighbours and multiply with dflip
    
    dH = -h*dflip - J *dflip*(np.sum(neighbours))
    
    return dH,rd_spin

#function that verifies the correct implementation of the energy_change() function

def cross_check_energy_change():
    """to check if the energy_change function is implemented correctly
    """

    print("Test: cross check of energy_change():")

    L = 20
    
    J = 1
    
    h = 0

    q = 0.5

    states = [create_state(q,L) for i in range(10)]

    success = True

    for state in states:

        site = random.randint(0,L**2-1)

        dH, rd_spin = energy_change(state,site,L,J,h)

        energy_old = energy(state, L, J, h)

        state[site] = -state[site]

        energy_new = energy(state, L, J, h)

        if (energy_new-energy_old) != dH:

            print("\t energy_new: ", energy_new)
            print("\t energy_old: ", energy_old)
            print("\t energy_diff: ",energy_new-energy_old)
            print("\t dH: ", dH)

            success = False
    
    print("\t success: ", success)
        
    
def calc_correlation(state, L):
    """calculates the correlation of spins along a horizontal cut

    Args:
        state (_type_): state to calculate the correlation for
        L (_type_): system size L

    Returns:
        _type_: array with correlation of spins (for a specific configuration)
    """

    # pick spin in the middle
    
    spin = state[int((L+1)*L/2)]
    
    index_row = int(divmod((L+1)*L/2,L)[0])

    correlation = []

    for i in range(index_row*L,index_row*L+L):

        correlation.append(spin*state[i])

    return correlation


# implements one entire monte carlo sweep

@jit(nopython=True)   
def metropolis_procedure(state,curr_t,L, J, h, sweeps):
    """Implements one single Metropolis step

    Args:
        state (_type_): state, the steps starts with
        curr_t (_type_): temperature
        L (_type_): system size L
        J (_type_): coupling strength
        h (_type_): magnetic field
        sweeps (_type_): boolean, which indicates if L*L number of spin flips should be attempted

    Returns:
        _type_: new state after (multiple) spin flips
    """
    new_state = np.copy(state)
    
    # one monte-carlo sweep
    
    number_sweeps = 1
    
    if sweeps:
        number_sweeps = L*L

    for i in range(number_sweeps):
        
        # pick random site
        
        rd_site = np.random.randint(0,L*L)

        # calculate energy change
        
        dH, rd_spin = energy_change(new_state,rd_site,L, J ,h)

        # if energy change is negative, flip spin
        # if not, draw a random number, and then maybe flip the spin
        
        if(dH>0):
            
            #draw random number
            p = np.random.random_sample()
            # if p < e^-dH/T, then flip the spin
            if(p<np.exp(-dH/curr_t)):
                #flip the spin
                new_state[rd_site]=-1*rd_spin
                
        else:
            new_state[rd_site]=-1*rd_spin

    return new_state

# implements the metropolis algorithm for an input temperature T, system size L
# after burn-in time t_0, sampling starts
# measurement of m,e,m^2 and e^2

def metropolis(T,state,L,M, J, h, time_series = False, correlation_result = False, burn_in = True, sweeps = True):
    """implements the Metropolis algorithm via multiple single Metropolis steps

    Args:
        T (_type_): temperature
        state (_type_): state, the simulation starts with
        L (_type_): system size
        M (_type_): number of samples
        J (_type_): coupling strength
        h (_type_): magnetic field
        time_series (bool, optional): if True, the time series of an observable is returned. Defaults to False.
        correlation_result (bool, optional): if True, the correlation of spins along a horizontal cut is returend. Defaults to False.
        burn_in (bool, optional): If True, a burn-in is first done to reach thermalization. Defaults to True.
        sweeps (bool, optional): If True, L*L spin flips are attempted in one single Metropolis step. Defaults to True.

    Returns:
        _type_: simulation results
    """
    energy_obs = []
    magnetisation_obs = []

    correlation = []
    
    # burn in phase
    if burn_in:
        for t in range(0,t0):
            
            state = metropolis_procedure(state,T,L, J, h, sweeps)
        
        print("burn-in phase finished")
    
    # sampling starts
    
    for t in range(0,M):
        
        # do one monte carlo sweep
        
        state = metropolis_procedure(state,T,L, J, h, sweeps)
        
        m = magnetisation(state)
        
        e = energy(state,L, J, h)
        
        magnetisation_obs.append(m/(L*L))
        
        energy_obs.append(e/(L*L))

        correlation.append(calc_correlation(state,L))
            
    correlation = np.array(correlation)
    
    # calc the average of the sums
    
    print("T = ",T," h = ",h," finished")
    
    # this below is a mess, but I had to do it this way
    
    if time_series:
        
        return (T, magnetisation_obs,np.power(magnetisation_obs,2), energy_obs, np.power(energy_obs,2))
    
    if correlation_result:

        return (T, np.mean(correlation, axis = 0))
    
    return (T, np.mean(magnetisation_obs), np.mean(np.power(magnetisation_obs,2)), np.mean(energy_obs), np.mean(np.power(energy_obs,2)))


# binning analysis as explained on the CP website
def binning_analysis(observables,k_max):
    """Binning analysis as suggested on the CP website

    Args:
        observables (_type_): sample of an observable
        k_max (_type_): maximal binning size

    Returns:
        _type_: array of error estimates for each binning size up to k_max
    """
    M = len(observables)
    
    error_estimate = []
    
    for k in range(1,k_max):
        
        M_k = np.floor(M/k)
        
        # reshape the array into M_k parts of length k
        
        observables_k = np.reshape(observables[0:int(k*M_k)],(-1,k))
        
        # calculate the mean along axis 1
        
        observables_k = np.mean(observables_k, axis = 1)
        
        error_estimate_k = np.std(observables_k)/np.sqrt(M_k)
        
        error_estimate.append(error_estimate_k)
        
    return error_estimate


# function that takes a sample of an observable an calculates the auto-correlation
# up to max_steps
def calc_auto_corr(sample, max_steps):
    """Calculates the auto-correlation for a given sample of an observable

    Args:
        sample (_type_): samples of an observable
        max_steps (_type_): max timestep to consider

    Returns:
        _type_: auto-correlation for each timestep
    """
    mean = np.mean(sample)
    var = np.var(sample)
    
    auto_corr = []
    
    for step in range(1,max_steps):
        
        corr = []
        
        for i in range(0,len(sample)-step):
            
            corr.append(sample[i]*sample[i+step])
        
        corr = np.mean(corr)
        
        auto_corr.append(corr)
    
    return (auto_corr-mean**2)/var
        
#################
# Implementation of the tasks
#################

def do_task_3_1_1():
    
    M = 20000
    L = 20
    
    # parameter for simulation
    J = 1
    h = 0
    
    state = create_state(1.0,L)
    
    temp_interval = np.arange(1,6,0.1)
    
    data = [metropolis(T,state,L,M, J, h) for T in temp_interval]
    
    np.savetxt("data/data_3_1_1.txt", data, header = "#T, m, m^2, e, e^2")
    
def do_task_3_1_2():
    
    M = 20000
    L = 40
    
    # parameter for simulation
    J = 1
    h = 0
    
    state = create_state(1.0,L)
    
    temp_interval = [2.5, 3.65, 5]
    
    data = [metropolis(T,state,L,M, J, h, time_series = True) for T in temp_interval]
    
    for data_T in data:
        
         auto_corr_m = calc_auto_corr(data_T[2],1000)
         auto_corr_e = calc_auto_corr(data_T[3],1000)
         
         auto_corr = np.column_stack((auto_corr_m,auto_corr_e))
         
         np.savetxt("data/data_3_1_2_T=" + str(data_T[0]) + ".txt", auto_corr, header = "# m e")
    
    #np.savetxt("data_3_1_1.txt", data, header = "#T, m^2, e, e^2")
    
def do_task_3_1_2_issue():
    
    M = 20000
    L = 40
    
    # parameter for simulation
    J = 1
    h = 0
    
    state = create_state(1.0,L)
    
    temp_interval = [2.5, 3.65, 5]
    
    data = [metropolis(T,state,L,M, J, h, time_series = True, sweeps=False) for T in temp_interval]
    
    for data_T in data:
        
         auto_corr_m = calc_auto_corr(data_T[2],1000)
         auto_corr_e = calc_auto_corr(data_T[3],1000)
         
         auto_corr = np.column_stack((auto_corr_m,auto_corr_e))
         
         np.savetxt("data/data_3_1_2_issue_T=" + str(data_T[0]) + ".txt", auto_corr, header = "# m e")
    
    #np.savetxt("data_3_1_1.txt", data, header = "#T, m^2, e, e^2")

def do_task_3_1_3():
    
    M = 50000
    
    L = 10
    
    # parameter for simulation
    J = 1
    h = 0
    
    state = create_state(1.0,L)
    
    temp_interval = [2.5, 3.65, 5]
    
    data = [metropolis(T,state,L,M, J, h, time_series = True, burn_in = False) for T in temp_interval]
    
    for data_T in data:
        
         error_estimates_m = binning_analysis(data_T[2],1000)
         error_estimates_e = binning_analysis(data_T[3],1000)
         
         auto_corr = np.column_stack((error_estimates_m,error_estimates_e))
         
         np.savetxt("data/data_3_1_3_L=" + str(L) +"_T=" + str(data_T[0]) + ".txt", auto_corr, header = "# m e")
         
    L = 20
    
    state = create_state(1.0,L)
    
    data = [metropolis(T,state,L,M, J, h, time_series = True, burn_in = False) for T in temp_interval]
    
    for data_T in data:
        
        error_estimates_m = binning_analysis(data_T[2],1000)
        error_estimates_e = binning_analysis(data_T[3],1000)
         
        auto_corr = np.column_stack((error_estimates_m,error_estimates_e))
         
        np.savetxt("data/data_3_1_3_L=" + str(L) +"_T=" + str(data_T[0]) + ".txt", auto_corr, header = "# m e")
         
def do_task_3_1_4():
    
    M = 20000
    
    # parameter for simulation
    J = 1
    h = 0
    
    range_L = [10, 20, 40, 50]
    
    for L in range_L:
        
        print("starting with L = ",L)
        
        state = create_state(1.0,L)
        
        temp_interval = np.arange(1,3,0.1)
        temp_interval = np.concatenate((temp_interval,np.arange(3,4,0.01)))
        temp_interval = np.concatenate((temp_interval,np.arange(4,6,0.1)))
        
        data = [metropolis(T,state,L,M, J, h) for T in temp_interval]
        
        np.savetxt("data/data_3_1_4_L=" + str(L) + ".txt", data, header = "#T, m, m^2, e, e^2")
        
def do_task_3_2_6():
         
    # parameter for simulation
    J = -1
    h = 0
     
    M = 50000
    
    L = 10
    
    state = create_state(1,L)
    
    temp_interval = [np.power(2,n/4) for n in range(-12,28+1)]
    
    data = [metropolis(T,state,L,M, J, h, time_series = True, burn_in=False, sweeps = False) for T in temp_interval]
    
    auto_corr_m = np.empty((999,0))
    auto_corr_e = np.empty((999,0))
    
    for data_T in data:
        
        error_estimates_m = binning_analysis(data_T[2],1000)
        error_estimates_e = binning_analysis(data_T[3],1000)
        
        auto_corr_m = np.column_stack((auto_corr_m,error_estimates_m))
        auto_corr_e = np.column_stack((auto_corr_e,error_estimates_e))
       
    np.savetxt("data/data_3_2_6_L=" + str(L) +"_m.txt", auto_corr_m)
    np.savetxt("data/data_3_2_6_L=" + str(L) +"_e.txt", auto_corr_e)
         
    L = 20
    
    state = create_state(1.0,L)
    
    data = [metropolis(T,state,L,M, J, h, time_series = True, burn_in=False, sweeps = False) for T in temp_interval]
    
    auto_corr_m = np.empty((999,0))
    auto_corr_e = np.empty((999,0))
    
    for data_T in data:
        
        error_estimates_m = binning_analysis(data_T[2],1000)
        error_estimates_e = binning_analysis(data_T[3],1000)
        
        auto_corr_m = np.column_stack((auto_corr_m,error_estimates_m))
        auto_corr_e = np.column_stack((auto_corr_e,error_estimates_e))
       
    np.savetxt("data/data_3_2_6_L=" + str(L) +"_m.txt", auto_corr_m)
    np.savetxt("data/data_3_2_6_L=" + str(L) +"_e.txt", auto_corr_e)
    
def do_task_3_2_7():
    
    M = 20000
    
    # parameter for simulation
    J = -1
    h = 0
    
    range_L = [10, 20, 40, 80]
    
    for L in range_L:
        
        print("starting with L = ",L)
        
        state = create_state(1.0,L)
        
        temp_interval = [np.power(2,n/4) for n in range(-12,28+1)]
        
        data = [metropolis(T,state,L,M, J, h) for T in temp_interval]
        
        np.savetxt("data/data_3_2_7_L=" + str(L) + ".txt", data, header = "#T, m, m^2, e, e^2")

def do_task_3_2_9():
    
    M = 20000
    
    # parameter for simulation
    J = -1
    h_interval = np.arange(0,10,0.05)
    
    L = 40
    T = 0.2
    
    state = create_state(1.0,L)
    data = [metropolis(T,state,L,M, J, h) for h in h_interval]
    
    h_interval = np.reshape(h_interval,(len(h_interval),1))
    
    data = np.column_stack((h_interval, data))
        
    np.savetxt("data/data_3_2_9.txt", data, header = "#T, m, m^2, e, e^2")
    
def do_task_3_2_10():

    M = 20000

    J = -1
    h = 0

    L = 80

    state = create_state(1.0,L)

    temp_interval = [1,0.1]

    data = [metropolis(T,state,L,M, J, h, correlation_result = True) for T in temp_interval]

    spin_corr = np.empty((L,0))

    for axis in data:

        spin_corr = np.column_stack((spin_corr,axis[1]))

    np.savetxt("data/data_3_2_10.txt", spin_corr)

    

#check the correct implementation of the energy_change()
#cross_check_energy_change()

do_task_3_1_2_issue()

#print(calc_auto_corr(np.arange(0,100,1),25))