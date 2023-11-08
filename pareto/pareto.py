'''
Method to take two equally-sized lists and return just the elements which lie 
on the Pareto frontier, sorted into order.
Default behaviour is to find the maximum for both X and Y, but the option is
available to specify maxX = False or maxY = False to find the minimum for either
or both of the parameters.
'''
def pareto_frontier_lists_x_y(Xs, Ys, maxX = True, maxY = True):
# Sort the list in either ascending or descending order of X
    myList = sorted([[Xs[i], Ys[i]] for i in range(len(Xs))], reverse=maxX)
# Start the Pareto frontier with the first value in the sorted list
    p_front = [myList[0]]    
# Loop through the sorted list
    for pair in myList[1:]:
        if maxY: 
            if pair[1] >= p_front[-1][1]: # Look for higher values of Y…
                p_front.append(pair) # … and add them to the Pareto frontier
        else:
            if pair[1] <= p_front[-1][1]: # Look for lower values of Y…
                p_front.append(pair) # … and add them to the Pareto frontier
# Turn resulting pairs back into a list of Xs and Ys
    p_frontX = [pair[0] for pair in p_front]
    p_frontY = [pair[1] for pair in p_front]
    return p_frontX, p_frontY

def pareto_fronts(Xs, Ys, maxX = False, maxY = False, nfronts=1):
    myList = sorted([[Xs[i], Ys[i]] for i in range(len(Xs))], reverse=maxX)
    # create nfronts
    fronts_x = []
    fronts_y = []

    for f in range(nfronts):
        if not myList: break
        # Start the Pareto frontier with the first value in the sorted list
        first = myList[0]
        p_front = [myList[0]]
        # Loop through the sorted list
        for pair in myList[1:]:
            if maxY:
                if pair[1] >= p_front[-1][1]: # Look for higher values of Y 
                    p_front.append(pair)     
                    myList.remove(pair)       
            else:
                if pair[1] <= p_front[-1][1]: # Look for lower values of Y…
                    p_front.append(pair) # … and add them to the Pareto frontier
                    myList.remove(pair)       
        # Turn resulting pairs back into a list of Xs and Ys
        fronts_x.append([pair[0] for pair in p_front])
        fronts_y.append([pair[1] for pair in p_front])
        # restart with remaining points:
        myList.remove(first)
        myList = sorted(myList, reverse=maxX)
  
    if nfronts-1:  
        return fronts_x, fronts_y
    else:
        return fronts_x[0], fronts_y[0]



def pareto_frontier_pairs(pairs, maxX = True, maxY = True):
# Sort the list in either ascending or descending order of X
    myList = sorted(pairs, reverse=maxX)
# Start the Pareto frontier with the first value in the sorted list
    p_front = [myList[0]]
# Loop through the sorted list
    for pair in myList[1:]:
        if maxY:
            if pair[1] >= p_front[-1][1]: # Look for higher values of Y…
                p_front.append(pair) # … and add them to the Pareto frontier
        else:
            if pair[1] <= p_front[-1][1]: # Look for lower values of Y…
                p_front.append(pair) # … and add them to the Pareto frontier
    return p_front
