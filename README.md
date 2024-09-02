# Classical Monte Carlo Simulation for Ising Models

## Overview

This project involves the use of classical Monte Carlo simulations to study the behavior of two-dimensional Ising models on different lattice structures. The main focus is on the phase transitions, magnetic properties, and thermodynamic quantities of the system using the Metropolis-Hastings algorithm. The models include both the square lattice and the triangular lattice Ising models. The project analyzes various physical quantities such as magnetization, energy, specific heat, and spin correlations to understand critical phenomena.

## Objectives

1. **Square Lattice Ising Model**: Simulate the 2D Ising model on a square lattice using the Metropolis algorithm to study phase transitions and critical phenomena.
2. **Triangular Lattice Ising Model**: Extend the simulation to a triangular lattice to explore the effects of lattice geometry on phase transitions and magnetic properties.
3. **Physical Quantity Calculation**: Compute physical quantities such as magnetization, energy, specific heat, and spin-spin correlations to analyze the system's behavior near the critical temperature.

## Key Components

- **`Ising_2d.py`**: Implements the 2D Ising model on a square lattice.
  - Functions to initialize the lattice state, calculate neighbors, and compute magnetization and energy.
  - Implements the Metropolis-Hastings algorithm to simulate the spin dynamics.
  - Calculates thermodynamic quantities such as energy, magnetization, specific heat, and susceptibility.

- **`Ising_triangular_2d.py`**: Extends the Ising model to a triangular lattice.
  - Functions for handling the additional nearest neighbors specific to the triangular lattice.
  - Similar Monte Carlo simulation framework as the square lattice, adapted for the triangular geometry.

## Theoretical Background

1. **Ising Model**: The Ising model is a mathematical model of ferromagnetism in statistical mechanics. It consists of discrete variables representing magnetic dipole moments of atomic spins that can be in one of two states (+1 or -1). Spins are arranged in a lattice, and each spin interacts with its nearest neighbors. The system is studied to understand phase transitions, such as the transition from a magnetically ordered to a disordered state at a critical temperature.

2. **Metropolis-Hastings Algorithm**: A Monte Carlo method used to obtain the equilibrium distribution of states in the system. The algorithm generates a sequence of states by proposing updates to the spin configurations and accepting or rejecting them based on a probability that satisfies the detailed balance condition.

## How to Run

1. **Requirements**:
   - Python 3.x
   - Numpy
   - Scipy
   - Matplotlib
   - Numba

2. **Execution**:
   - To simulate the 2D Ising model on a square lattice:
     ```bash
     python Ising_2d.py
     ```
   - To run the simulation on a triangular lattice:
     ```bash
     python Ising_triangular_2d.py
     ```

## Results

- **Square Lattice Model**:
  - The simulations compute and plot the magnetization, energy, specific heat, and susceptibility as functions of temperature.
  - Results show a sharp phase transition at the critical temperature \( T_c \approx 2.269 \).

- **Triangular Lattice Model**:
  - Similar physical quantities are computed for the triangular lattice, and differences due to the additional nearest neighbors are analyzed.
  - The impact of lattice geometry on critical behavior and magnetic properties is studied.

## Figures and Analysis

- **Figure 1 to 7**: Show various aspects of the Ising model on square and triangular lattices, including magnetization, energy, autocorrelation functions, specific heat, and spin correlations.
- **Figure 8 to 13**: Illustrate the behavior of physical quantities such as specific heat and magnetization across different temperatures and system sizes, showing the phase transitions and critical phenomena.

## Conclusion

This project provides an in-depth study of the 2D Ising model on different lattice structures using Monte Carlo simulations. The results highlight the influence of lattice geometry on the phase transition and critical behavior of the system. The simulations confirm the expected theoretical predictions, such as the critical temperature and specific heat scaling, while also offering insights into the differences between square and triangular lattices.

## Future Work

- Extend the simulations to three-dimensional lattices to explore more complex critical phenomena.
- Incorporate external magnetic fields and study their effects on phase transitions.
- Use larger lattice sizes and improved numerical methods to reduce finite-size effects and improve accuracy.
