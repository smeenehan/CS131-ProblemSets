import numpy as np

def circle(x,y,R,nPoints):
    center = np.array([x,y])
    theta = np.linspace(0,2*np.pi,num=nPoints)
    return center+R*np.array([np.cos(theta),np.sin(theta)]).T

def fitCircle(points):
    """
    Inputs:
    points: Nx2 NumPy array of (x,y) coordinate

    Outputs:
    [cx,cy,R], best fit to circle center and radius
    """
    (nPoints,nCoords) = np.shape(points)
    if nPoints<3:
        print("Error: You need at least 3 points to fit a circle")
        return []
    b = points[:,0]**2+points[:,1]**2
    A = np.concatenate((points,np.ones((nPoints,1))),1)
    results = np.linalg.lstsq(A,b)
    x = results[0]
    cx,cy = x[0]/2,x[1]/2
    R = np.sqrt(x[2]+cx**2+cy**2)
    return [cx,cy,R]

def genData(nPoints,sigma=0.1,outlierFraction=0):
    """
    Inputs:
    nPoints: # of points desired
    sigma: standard deviation of random Gaussian noise,
    outlierFraction: fraction of points drawn from uniform random
     distribution

    Outputs:
    [data,x,y,R]
    data: nPoints x 2 NumPy array of points on the circle (w/noise + outliers)
    x,y: true circle center
    R: true circle radius
    """
    nOutliers = int(np.round(outlierFraction*nPoints))
    nCircle = nPoints-nOutliers

    x = 1-2*np.random.random()
    y = 1-2*np.random.random()
    R = 1+4*np.random.random()

    outlierData = np.array([x,y])+8*R*(np.random.random((nOutliers,2))-0.3)
    circleData = circle(x,y,R,nCircle)+sigma*np.random.randn(nCircle,2)

    data = np.concatenate((outlierData,circleData))
    np.random.shuffle(data)
    return [data,x,y,R]

def randomlySplitData(data,splitSize):
    idx = np.random.permutation(np.shape(data)[0])
    return [data[idx[:splitSize],:],data[idx[splitSize:],:]]

def computeErrors(x,y,R,data):
    return abs(np.linalg.norm(data-[x,y],axis=1)**2-R**2)

def RANSAC(data,maxIter,maxInlierError,fitThresh):
    """
    Inputs:
    data: Nx2 NumPy array of points to fit
    maxIter: number of RANSAC iterations to run
    maxInlierError: max error used to determine if a non-seed point is an
     inlier of the model. Error is measured as abs(dist**2-R**2)
    fitThresh: threshold for deciding whether a model is good; a minimum
     of fitThresh non-seed points must be inliers

    Outputs:
    [x,y,R] best fit center and radius
    """
    seedSize = 3
    [x,y,R,bestError] = [0,0,0,np.inf]

    for idx in range(maxIter):
        [seedSet,nonSeedSet] = randomlySplitData(data,seedSize)
        [cx,cy,cR] = fitCircle(seedSet)
        errors = computeErrors(cx,cy,cR,nonSeedSet)
        inliers = nonSeedSet[errors<=maxInlierError,:]
        if np.size(inliers) >= fitThresh:
            inliers = np.append(seedSet,inliers,axis=0)
            [cx,cy,cR] = fitCircle(inliers)
            totalError = np.sum(computeErrors(cx,cy,cR,inliers))
            if totalError < bestError:
                [x,y,R,bestError] = [cx,cy,cR,totalError]
    return [x,y,R]
