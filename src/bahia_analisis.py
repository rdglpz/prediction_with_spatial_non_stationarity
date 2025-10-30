import numpy as np

def constructTocSeries(df, 
                       delta, 
                       points, 
                       year_init = 1985, 
                       year_end = 2020
                      ):

    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_curve
    import matplotlib.pyplot as plt
    from pytoc import TOC, TOC_painter
    
    directory = "databases_interval_comparison_ba/"
    prefix = "dist_change_bahia_"
    
    interval_labels = [str(y) + "_" + str(y + delta) for y in 
                       np.arange(year_init, year_end - delta + 1)]
    
    TOCs = []
    
    for i, s in enumerate(interval_labels):
        
        path = directory + prefix + s + ".csv"
        print(path)
        
        db = pd.read_csv(path)
        
        #tomamos las dos primeras columnas que son las coordenadas
        #distancia
        pair = np.array(db.iloc[:, [0, 1, -3, 2, -2,]])
        
        rank, y = get_rank_y_pair(pair)
        
        
        #reverse_rank shoudl be the inverse of distance rank
        #closer cells must have highest rank
        reverse_rank =  np.max(rank) - rank
        
        #indices que apuntan a valores de mayor a menor valor
        rank_sorted = np.flip(np.argsort(reverse_rank.flatten()))
        rank_descend_ss = reverse_rank[rank_sorted]
        y_aligned = y[rank_sorted]
        
        six = np.arange(0, len(rank_sorted), 1000)
        division = np.linspace(0, 100, 32)
        perc = np.percentile(reverse_rank, division)
        uranks = np.unique(perc)
        
        booleanArray = y_aligned
        indexArray = rank_descend_ss
        
        
        #print("indexArray,", indexArray)
        
      #  plt.plot(indexArray)
       # plt.show()
        
        thresholds = np.flip(uranks)
        
        #print("thresholds,", thresholds)
       # plt.plot(thresholds)
       # plt.show()
        
        TOC_1 = TOC(booleanArray, indexArray, thresholds)
        TOCs.append(TOC_1)
        #painter = TOC_painter(TOC_list=[TOC_1], 
        #                      index_names=['distance'], 
        #                      color_list=['r'], 
        #                      marker_list=['^'], 
        #                      line_list=['-'], 
        #                      boolUniform=True, boolCorrectCorner=True)
        #painter.paint()
        
    return TOCs
        
        
        
        
        
    
    

def comparisonsTimeSeries(df, delta, points, year_init = 1985, year_end = 2020  ):
    
    import numpy as np
    import pandas as pd
    
    #list of interval labels e.g., 1985_1987
    interval_labels = [str(y) + "_" + str(y + delta) for y in 
                       np.arange(year_init, year_end - delta + 1)]
    

    
    
    
    
    print("Saving table of coordinates, Initial distances, \
        changes in an interval from ")
    for i, s in enumerate(interval_labels):
        
        losses = pd.DataFrame(points)
    
        map1 = np.array(df[i + 2])
        map2 = np.array(df[i + 2 + delta])
        
        #V is the vector of change
        
        
        
        V = change(map1, map2)
        
        #print("len V", len(V))
        #print("len ", len(df.iloc[:,-3]))
        
        
        
        losses["distances"] = map1
        losses["slope"] = np.array(df.iloc[:,-3])
        losses[s] = V
        
        print("saving:")
        
        name = "dist_change_bahia_" + s + ".csv"
        print(name)
        
        losses.to_csv(name)
        
        
def change(map1, map2):
    
    
    
    #0 is forest 
    #1 is non-forest
    
    
    #persistence_non-forest, barren -> barren
    pers_non_forest = ((map1==0) & (map2==0))*0
    
    #lost non-forest (gain forest), barren to forest
    lost = ((map1==0) & (map2!=0))*1
    
    
    #----These transitions are of interest
    
    #gain non_forest, forest to barren
    gain = ((map1!=0) & (map2==0))*2
    
    #persistence forest
    pers_forest = ((map1!=0) & (map2!=0))*3
    
    return pers_non_forest + lost + gain + pers_forest
    
    
    
def invert_rank(rank):
    
    return np.max(rank)-rank


def get_rank_y_pair(ch1, dv = 4):
    
    import numpy as np
    
    """
    0: num de fila
    1: coordenada en puntos 
    2: distancia a no bosque
    3: pendiente 
    4: persistencia (3) o deforestacion (4)
    
     #gain non_forest, forest to barren
    gain = ((map1!=0) & (map2==0))*2
    
    #persistence forest
    pers_forest = ((map1!=0) & (map2!=0))*3
    
    
    """
    
    
    #regresamos solo los indices relacionados con gain non forest (forest -> barren): 2,  persistent forest, forest -> forest : 3
    #
    ix = np.argwhere(1*(ch1[:, dv]==2) + 1*(ch1[:, dv]==3))
    rank = ch1[ix, 0]
    y = ((ch1[ix, dv]==2)*1).flatten()
    
    return rank, y

def mapComparison(t):
    """
    0: barren -> barren: gray
    1: barren -> forest: 
    2: forest -> forest: green
    3: forest -> barren: red
    """
    
    if t[0]==0 and t[1]==0:
        return 0
    
    if t[0]==0 and t[1]!=0:
        return 1
    
    if t[0]!=0 and t[1]!=0:
        return 2
    
    if t[0]!=0 and t[1]==0:
        return 3

def generate_dataset(X, Y, Y_patch, h, mask):
    """
    X: is a sequence of images of shape (height, width, sequence) from [year-L, year)
    Y: Y is a variable that indicates the prescence for the next time year
    Y_patch: is the complete frame of the next time year


    """
    
    import numpy as np
    
    L = np.shape(X)[2]
    width = h*2 + 1
    height = h*2 + 1
    
    #number of matrices with (h*2 + 1)x(h*2 + 1) dimension with L bands
    nm = len(range(h, np.shape(X)[0]-h-1)) * len(range(h, np.shape(X)[1]-h-1))
    
    XX = np.zeros((nm, height, width, L))
    YY = np.zeros(nm)
    YY_patch = np.zeros((nm, height, width))
    flatten_mask = np.zeros(nm)
    non_forest_last_step = np.zeros(nm)
    
    k = 0

    for i in range(h, np.shape(X)[0]-h-1):
        for j in range(h, np.shape(X)[1]-h-1):
        
            # XX has k subimages of heigh x width x lags
            XX[k, :, :, :] = (X[i-h:i+h+1, j-h:j+h+1, :]==0)*1

            # if the center in the next lag is deforested then it is equal to 1
            YY[k] = Y[i,j]*1
            
            YY_patch[k] = (Y_patch[i-h:i+h+1, j-h:j+h+1]==0)*1
            
            flatten_mask[k] = mask[i,j]
            
            non_forest_last_step[k] = (X[i, j, -1]!=0) 
            k = k+1
   
    ix = np.argwhere((flatten_mask==1) & (non_forest_last_step==1))
            
    return XX[ix], YY[ix], YY_patch[ix]
    
def generate_canonical_dataset(sequence_of_images, YEAR, LAGS, HALF_SIZE, mask):
    """
    YEAR: relative year, 0 is the first year assosiated with the absolute year 1985
    LAGS: number of past observations
    HALF_SIZE: the half of the squared size
    """
    
    D2NF = np.copy(sequence_of_images)
    
    NON_FOREST = 0
    h = HALF_SIZE
    bahia_mask = mask
    
    X = D2NF[:, :, YEAR-LAGS:YEAR ]
    Y = (D2NF[:, :, YEAR-1]!=NON_FOREST) & (D2NF[:, :, YEAR]==NON_FOREST) 
    Y_patch = D2NF[:, :, YEAR]
    X_tr, Y_tr, Y_tr_patch = generate_dataset(X, Y, Y_patch, h, bahia_mask)
    
    return X_tr, Y_tr, Y_tr_patch
    