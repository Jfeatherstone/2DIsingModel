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

