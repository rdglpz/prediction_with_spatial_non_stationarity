
import numpy as np
import georaster



def makeGrid(rows, columns, block_size = 100):

    total_size_x = rows
    total_size_y = columns

    # Create an empty array of zeros
    grid = np.zeros((total_size_x, total_size_y), dtype = int)

    # Fill the array with grid pattern
    for i in range(0, total_size_x, block_size):
        for j in range(0, total_size_y, block_size):
            grid[i : i+block_size, j :j+block_size] = (i // block_size) * ((total_size_y // block_size)+1) + (j // block_size)

    return grid


def readMask(dir =  "../data/processed/gtiff/mask_bahia_gtiff"):

    mask = georaster.MultiBandRaster(dir)
    bahia_mask = (mask.r==255)!=True
    bahia_mask = bahia_mask[:,:,0]
    bahia_mask = np.where(bahia_mask==0, np.nan, bahia_mask)

    return bahia_mask


def loadMapinNumpy(directory = "../data/processed/gtiff/"):

    recentdefpath = "recent_deforestation_density/deforestation1985_1986_dens_geotiff"
    loss_distpath = "/loss_dist/loss{a}_dist_geotiff"

    yinit = 1985
    yend = 2020

    bahia_mask = readMask()
    img = georaster.MultiBandRaster(directory+recentdefpath)
    sh = np.shape(img.r)

    #the shape of the tensor containing all the 36 deforestation images
    tensor = (sh[0], sh[1], yend-yinit+1)

    #initializing the 3D tensors that will contain all the temporal maps
    # hast all the D2NF
    D2NF = np.zeros(tensor)

    years = np.arange(yinit, yend + 1) 
    for i, y in enumerate(years):

        path_to_d2nf = "../data/processed/gtiff/loss_dist/loss{a}_dist_geotiff".format(a = y)
        M = georaster.MultiBandRaster(path_to_d2nf)
        D2NF[:, :, i] = M.r[:, :, 0]*bahia_mask


    return D2NF

def makeGrid(rows, columns, block_size = 100):

    total_size_x = rows
    total_size_y = columns

    # Create an empty array of zeros
    grid = np.zeros((total_size_x, total_size_y), dtype = int)

    # Fill the array with grid pattern
    for i in range(0, total_size_x, block_size):
        for j in range(0, total_size_y, block_size):
            grid[i : i+block_size, j :j+block_size] = (i // block_size) * ((total_size_y // block_size)+1) + (j // block_size)

    return grid

def generateTrainingDataset(DF, F, predictionHorizon = 1, LAGS = 1, Psize = 1, year0= 1985, year1= 2015):

    
    
    
    H = predictionHorizon
    interval_size = len(DF.columns)
    GriddedDeforest = DF[year0].to_numpy()
    GD = np.reshape(GriddedDeforest, (F, -1))
    shape = np.append([interval_size] , [np.array(GD.shape)+Psize*2])
    PADDING = np.zeros(shape)

    #Generate padding for forest 

    for i, y in enumerate(range(year0, year1+1)): 
    
        GriddedDeforest = DF[y].to_numpy()
        GD = np.reshape(GriddedDeforest, (F, -1))
        PADDING[i, Psize:Psize+GD.shape[0], Psize:Psize+GD.shape[1]] = np.ones(GD.shape)*GD

    number_of_years = PADDING.shape[2]    
    RA = [i for i in range(LAGS, number_of_years-H)]
    RV = [i for i in range(Psize, Psize+GD.shape[0])]
    RH = [i for i in range(Psize, Psize+GD.shape[1])]

    #años x renglones x rows
    iterations = iter.product(RA, RV, RH)

    X = list([])
    Y = list([])

    for i in iterations:
        
        x = i[0]
        y = i[1]
        z = i[2]

        Xnext = PADDING[x+H, y-Psize:y+Psize+1, z-Psize:z+Psize+1]

        if True:

            XX = PADDING[x-LAGS:x, y-Psize:y+Psize+1, z-Psize:z+Psize+1]
            X.append(XX.T)
            Y.append(Xnext[Psize,Psize])
    return X,Y










