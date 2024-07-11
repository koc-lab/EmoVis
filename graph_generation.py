from imports import *

def project_vector(x,y):
    #projects x onto y
    return np.dot(x, y) / np.linalg.norm(y)

def vector_hvg(series, timeLine=None):
    # series is the data vector to be transformed
    if timeLine == None: timeLine = range(len(series))
    # Get length of input series

    ## series gets a multivariate time series. y(t) is n dimensional

    L = len(series)
    # initialise output
    all_visible = []

    for i in range(L-1):
        node_visible = []
        ya_embedding = series[i]
        ya = project_vector(ya_embedding,ya_embedding)

        ta = timeLine[i]
        for j in range(i+1,L):

            yb_embedding = series[j]
            yb = project_vector(yb_embedding,ya_embedding)
            tb = timeLine[j]

            yc_embedding = series[i+1:j]
            tc = timeLine[i+1:j]
            yc = project_vector(yc_embedding,ya_embedding)

            if all( yc[k] < min(ya,yb) for k in range(len(yc)) ):
                node_visible.append(tb)
            elif all( yc[k] >= max(ya,yb) for k in range(len(yc)) ):
                break

        if len(node_visible)>0 : all_visible.append([ta, node_visible])

    return all_visible