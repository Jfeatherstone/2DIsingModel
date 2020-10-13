import numpy as np

"""
This is an implementation of the Wolff spin flip update system
that characteristically does not loose efficiency around criticality.

Original publication:
https://doi.org/10.1103/PhysRevLett.62.361

This case will consider the 2D Ising model, which in the notation of
the above paper, means O(n) = 1, and R(r) just flips the sign of \sigma_x

This also means that the method is mostly simplified to the Swendsen and Wang
method cited in the above paper.

Other notes:
B is the inverse coupling temperature (Beta in the paper) and is represented with 1/T
Method requries periodic boundary conditions
Method only works with h=0 (no external magnetic field)

Overview of method:
a. Choose a random lattice site [i,j] as the first point in the cluster (x)
b. Flip the spin at [i,j]
c. Visit all nearest neighbors ([i+1,j], [i-1,j], [i,j+1], [i,j-1]) (y)
    Flip the spin of with the following probability:
    1 - exp(-2*B) if x == y; 0 otherwise
d. Record each visited point, and recursively make each y the new x, until no
    more sites get flipped; do not repeat sites that have already been visited
"""
def singleClusterFlip(isingMap, beta):
    """
    This method defines a single cluster flip on a given state isingMap

    isingMap should be an NxN array populated with either 1 or -1

    previouslyVisitedPoints can be excluded from the function call
    since that will be used to recurse

    Similarly, x can be left as None for non-recursive function call
    """

    # Make sure that the size of the array is correct
    if not len(np.shape(isingMap)) == 2:
        raise Exception(f'Invalid isingMap passed to singleClusterFlip; incorrect dimensions ({len(np.shape(isingMap))}), should be 2')

    N,M = np.shape(isingMap)

    if not N == M:
        raise Exception(f'Invalid isingMap passed to singleClusterFlip; shape is {np.shape(isingMap)} but dimensions should be equal')

    # Part a
    # We choose a random point x = (i,j) within [N,N]
    x = tuple(np.random.randint(0, N, size=2))

    # Part b
    # Flip the spin at x
    isingMap[x] = -isingMap[x]

    # Note that we've visited it
    previouslyVisitedPoints = []
    previouslyVisitedPoints.append(x)

    # Any empty list that will hold the [x, y] pairs we need to check
    pairsToCheck = []
    
    # Part c
    # Visit each nearest neighbor
    # Note that you cannot add tuples in Python, so we have to convert to
    # numpy arrays and then back to tuples
    # np.mod takes care of periodic boundary conditions
    nearestNeighborDirections = [[0, 1], [0, -1], [1, 0], [-1, 0]]
    yArr = [tuple(np.mod(np.array(x) + d, [N,N])) for d in nearestNeighborDirections]
  
    # Create pairs of x and y
    initialPairs = [[x, y] for y in yArr]
  
    # And add them to the running list
    for p in initialPairs:
        pairsToCheck.append(p)

    while len(pairsToCheck) > 0:
        x = pairsToCheck[0][0]
        y = pairsToCheck[0][1]

        # Skip if they have anitparallel spins
        if not isingMap[y] == isingMap[x]:
            pairsToCheck.pop(0)
            continue
       
        # Skip if we have already checked it
        if y in previouslyVisitedPoints:
            pairsToCheck.pop(0)
            continue

        r = np.random.uniform(0, 1)
        if r <= 1 - np.exp(-2*beta):
            # Flip the spin
            isingMap[y] = - isingMap[y]

            # Add it to our list of points and remove it from the list of ones to check
            previouslyVisitedPoints.append(y)
            pairsToCheck.pop(0)

            # And add all of the neighbors that haven't already been checked
            newPoints = [tuple(np.mod(np.array(y) + d, [N,N])) for d in nearestNeighborDirections]
            newPairs = [[y, z] for z in newPoints]
            for i in range(len(newPairs)):
                if not newPoints[i] in previouslyVisitedPoints:
                    pairsToCheck.append(newPairs[i])
        else:
            # If we don't flip, we still get rid of the site
            pairsToCheck.pop(0)

#    # Now actually check each one
#    for y in yArr:
#        # Skip if they have anitparallel spins
#        if not isingMap[y] == isingMap[x]:
#            continue
#       
#        # Skip if we have already checked it
#        if y in previouslyVisitedPoints:
#            continue
#
#        r = np.random.uniform(0, 1)
#        if r <= 1 - np.exp(-2*beta):
#            # Recurse on this point
#            singleClusterFlip(isingMap, beta, previouslyVisitedPoints, y)
        

"""
This is the original method to take a Monte-Carlo step, by randomly sampling a single site

I've included it to test the Wolff method above
"""
def singleSiteFlip(isingMap, T, h):
    """
    Flips a single site on the model according to probability involving
    the change in energy and the temperature
    
    Returns True if a site is flipped, False if not
    """
    N,M = np.shape(isingMap)

    x = tuple(np.random.randint([0, 0], [N, M], size=2))

    dE = deltaE(isingMap, x, h)
    acceptanceProb = min(np.exp(-dE/T), 1)

    r = np.random.uniform(0, 1)
    if r <= acceptanceProb:
        isingMap[x] = -isingMap[x]
        return True

    return False

# This defines the energy density of a given ising map (set of spins)
# Mostly taken from in class, but adapted to 2D
def energyDensity(isingMap, h):
    """
    Returns the energy density of a given spin setup

    Assumes J = 1
    """
    # This technically works for non-square setups, but that will likely never
    # happen
    N,M = np.shape(isingMap)

    # Collect the energy of each nearest neighbor
    # The easiest way to not double count any cells is to
    # go row by row and then column by column, as opposed to
    # doing both for a given point at once
    interactionEnergy = 0
   
    # Unlike in class, I'll account for the negative up front
    # Add horizontal interaction energies
    interactionEnergy += np.sum([-isingMap[:,i] * isingMap[:,(i+1)%M] for i in range(M)])
    # Add vertical interaction energies
    interactionEnergy += np.sum([-isingMap[j,:] * isingMap[(j+1)%N,:] for j in range(N)])

    # The magnetic contribution is easy
    magneticEnergy = -h * np.sum(isingMap)

    return (interactionEnergy + magneticEnergy)/(N*M)

# This defines the change in energy if we were to flip a single spin
# Expanded version of what we did in class
def deltaE(isingMap, x, h):
    # Grab sizes
    N,M = np.shape(isingMap)
    
    # Taken directly from the singleClusterFlip method (see there for more info)
    nearestNeighborDirections = [[0, 1], [0, -1], [1, 0], [-1, 0]]
    yArr = [tuple(np.mod(np.array(x) + d, [N,M])) for d in nearestNeighborDirections]

    # The factors of 2 come from the spin going from 1 -> -1 (or vice versa)
    interactionEnergy = 2 * np.sum([isingMap[x] * isingMap[y] for y in yArr])
    magneticEnergy = 2 * h * isingMap[x]
    return interactionEnergy + magneticEnergy


# Note that we do not divide by N^2 here
def magnetization(isingMap):
    return np.sum(isingMap)

# Some initialization methods
# Pretty basic, very similar to what we did in class
def initializeHot(N):
    isingMap = np.random.randint(0, 2, size=(N,N))
    isingMap[isingMap == 0] = -1
    return isingMap

def initializeCold(N):
    isingMap = np.zeros([N,N]) + 1
    return isingMap

# This is used for calculating the magnetization
def statisticalBootstrap(arr, func, M):
    # Select N random samples from the arr and take the value of the function
    # for each sampling
    O = np.array([func(np.random.choice(arr, size=len(arr))) for i in range(M)])

    # Return the average of O and the standard deviation
    #return [np.mean(O), np.sqrt(np.var(O))]
    # jk, we don't care about error so we only need the mean
    return np.mean(O)
