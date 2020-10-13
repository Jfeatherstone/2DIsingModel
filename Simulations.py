"""
This file contains methods that involve more complete Ising simulations when
compared the Ising.py. The latter file defines the base methods to 
manipulate and extract basic info from a given model, while here we go
more in depth.
"""

import numpy as np
import Ising

# This method calculates the magnetic susceptability and the specific heat for a given system size (N)
# and temperature. To calculate the magnetic susceptability, we find
# <M^2> - <M>^2, which requires lots of time steps and the statistical bootstrap
# method. Specific heat is found similarly as <E^2> - <E>^2
def calculateObservables(N, temp, thermalizationSteps=50, simulationSteps=100, skipSweeps=5, bootstrapM=500):
    """
    N should be the system length

    temp is the temperature of the model

    thermizationSteps is the number of steps to evaluate before
    calculating the susceptability. This is an estimate based on the
    results shown in the ThermalizationTest.ipynb file. Since the cluster
    flip method is rather fast, we can increase this without causing too
    much of a slowdown

    simulationSteps is the number of steps to use to calculate the susceptability

    returns [specific heat (C), susceptability (chi)]
    """

    # Where we will store the magnetizations and energies
    magnetizationArr = np.zeros(simulationSteps)
    energyArr = np.zeros(simulationSteps)

    # This isn't really a variable since the Wolff method requires h = 0,
    # but I'll define it to make the formulas be more verbose
    # That being said ****DO NOT CHANGE THIS****
    h = 0

    # Generate our initial conditions
    # Use cold start, but it shouldn't matter since we thermalize
    isingMap = Ising.initializeCold(N)
   
    # Thermalize
    for i in range(thermalizationSteps):
       Ising.singleClusterFlip(isingMap, 1/temp) 

    # Now we'll actually record data
    for i in range(simulationSteps):
        # Note that this magnetization is not the magnitization per site, so
        # we have to divide by a factor of N later on
        magnetizationArr[i] = Ising.magnetization(isingMap)

        # This is indeed the energy density, and therefore we multiply by a
        # factor of N later on as opposed to the magnetization above
        energyArr[i] = Ising.energyDensity(isingMap, h)

        # Skip a few steps to reduce the correlation between successive measurements
        for j in range(skipSweeps):
            Ising.singleClusterFlip(isingMap, 1/temp)


    # The variance in the magnetization should be proportional to the susceptability
    magnetizationVariance = Ising.statisticalBootstrap(magnetizationArr, np.var, bootstrapM)
    # The variance in the energy should be proportional to the specific heat
    energyVariance = Ising.statisticalBootstrap(energyArr, np.var, bootstrapM)

    # These constants out front are questionable, but in general were taken from
    # https://www.maths.tcd.ie/~bouracha/reports/2-dimensional_ising_model.pdf
    # Because of this uncertainty, we opted to solve for alpha and nu, since they
    # don't depend on the magnitude of the peaks, only their positions
    chi = (1/N**2) * 1/temp * magnetizationVariance
    C = N**2 * (1/temp)**2 * energyVariance

    return [C, chi]
