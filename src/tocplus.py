#!/usr/bin/env python3

#Object Oriented implementation for computing the
#TOC= Total Operating Characteristic Curve
#Author: S. Ivvan Valdez and R. Lopez-Farias
#Centro de Investigación en Ciencias de Información Geoespacial AC
#Querétaro, México.

import numpy as np
import copy
import matplotlib.pyplot as plt
import gc

from matplotlib.ticker import MaxNLocator, FuncFormatter, AutoMinorLocator
import matplotlib.ticker as ticker

#Object to store a TOC curve
class TOC:    
    """
    
    This class implements the Total Operating Characteristic Curve. The TOC computed from a rankl and groundtruth using this instantiation is named **standard TOC** trought the document.
    
    :param rank: The class is instantiated with the optional parameter ``rank`` that is a numpy array of a predicting feature.
    
    :param groundtruth: The class is instantiated with the optional parameter ``groundtruth`` that is a numpy array of binary labels (0,1). 
    
    :param sortw: 1 by default, the data is sorted using the formula: 1e9*rank+sortw*groundtruth, that means that data with the highest class value is sorted first.
    
    :ivar kind: A string with the value 'None' by default, it indicates the kind of TOC, for instance: TOC, normalized, vector, density and qspline. The ``kind`` attibute is mainly used for plotting inside the class.
    
    :return: The class instance, if ``rank`` and ``grountruth`` are given it computes the TOC, and the ``kind`` is TOC, otherwise it is an empty class. Some methods of the TOC class return the other kinds of TOC, suich as density, noprmalize, etc. 
    
    :rtype: ``TOC``
    
    """   
    
    
    area = 0
    
    """
    
    Area under the curve of the TOC.
    
    """
    
    
    areaRatio = 0    
    """
    
    areaRatio with respect to the parallelogram. If the maximum TOC area is that of the parallelogram, hence this is the ratio of the TOC area inside the parallelogram divided by the parallelogram arena. 
    Notice that the area inside the parallelogram is usually different from the area under the curve, it is computed substracting from the ``area`` the triangle in the left side of the parallelogram. 
    
    """
    pararea = 0    
    """
    
    parallelogram area.  
    
    """
    

    basenpos = 0
    """
    
    basenpos preserve the npos value from the standard TOC when it is normalized. Nevertheless it is modified by resample.
    
    """
    
    basentppfp = 0
    """
    
    Some TOCs result from applying operations to an standard TOC, in such a case the ntppfp could change int the resulting TOC nevertheless the basentppfp is copied from the standard TOC.
    For instance, ntppfp is 1 in the normalized TOC, but the basetppfp preserves the value from the standard TOC.
    
    """
        
    
    kind='None' 
    """
    
        The ``kind`` attribute indicates the type of TOC curve, that is to say, it is **"TOC"** kind if the *x* axes presents the true positives plus false positives count, and the *y* axes presents the true-positive count.
        It is a **"Vector""** kind if the curve is a representation of the original TOC, that is to say it stores *nvector* vectorial means of the original TOC data.
    
    """
    
    indices=None
    """
    
        This attibute stores the indices fo the sorted rank. Storing the indices is useful to save computational time when converting a rank array to a probability value.
    
    """
    
    ndata=0
    """
        This attribute stores the number of data in the arrays of the class,  notice that the number of positives (``npos``) or other counts are altered when interpolations, normalization or vectorization of the TOC is applied, while ``ndata`` stores the length of the data arrays independent of the mentioned counts.
    
    """
    
    ntppfp=0
    """
    This is the count fo true positives plus false positives. Notice that this number could be different from the number of data, for instance, normalized, vector, resampled or density TOCs could have a different ndata and ntppfp value.
    """

    npos=0
    """
    
    Similar to ntpfp, this variable stores the number of positive data (1 valued data), and also it is lost in the normalized TOC ( it is 1 in th enormalized TOC).
    
    """
    
    PDataProp=0    
    """
    
    Proportion of positive data in the data. The purpose is to maintain the proportion of class 1 data, hence to preserve the knowlkedege about data imbalance and proportion of calsses even if the TOC is normalized.
    
    """
    
    
    TPplusFP=None
    """
    A numpy array with the sum of true positives plus false positives.
    
    """
    
    TP=None
    """
    
    A numpy array with the sum of true positives
    
    """
    
    thresholds=None
    """
    
    A numpy array with thresholds of the TOC, they are computed using the ranks, that is to say, in the standard TOC most of the ranks are equal to the thresholds.
    
    """
    idiscontinuous = None
    isDiscontinuous = None
    h = None
    
    def __init__(self, rank = np.array([]), groundtruth = np.array([]), sortw = 1.0):
        """
        Constructor of the standard TOC
    
        """
   
        #validación de las tuplas (rank, grountruth)
        if (len(rank)!=0 and len(groundtruth)!=0 and len(rank)==len(groundtruth)):
            
            self.maxr = np.max(rank)
            self.maxgt = np.max(groundtruth)
            self.minr = np.min(rank)
            self.mingt = np.min(groundtruth)
            
            #rank normalizado
            trank = 1e9*(rank-self.minr)/(self.maxr-self.minr)+sortw*(groundtruth-self.mingt)/(self.maxgt-self.mingt)
            
            #Sorting the classification rank and getting the indices
            #indices = sorted(range(len(trank)), key=lambda index: trank[index], reverse=True)
            
            #aquí use lexsort que te da los indices de una lista ordenada de menor a mayor priorizando trank y despues groundtruth
            indices = np.lexsort((groundtruth, rank))
            
            #Storing the indices of the sorted rank
            self.indices = indices
            
            #Data size, this is the total number of samples
            self.ndata = n = len(rank)+1

            #Number of true positives plus true negatives, for representations differente than TOC kind the ndata is different to the ntppfp
            self.ntppfp = n = len(rank) 
            self.basentppfp = n-1
            #This is the number of class 1 in the input data
            self.npos=P=sum(groundtruth == 1)
            self.basenpos=P
            
            #True positives plus false positives (It is a simplified method)
            self.TPplusFP = np.arange(n+1)
            
            #True positives
            self.TP = np.append(0, np.cumsum(groundtruth[indices]))
            
            #Thresholds
            self.thresholds = np.append(rank[indices[0]], rank[indices]) #Ensuring than rank is positive
            self.rank = rank
            
            #[rodrigo] Select the the thresholds with the maxiumum acumulated TPFP
            
            sorted_rank = rank[indices][::-1]
            j, i = np.unique(sorted_rank, return_index = True)
            
            #we make a zero array with the same size of the data
            res = np.zeros_like(rank, dtype = bool)
            
            #we mark as true the specific non repeated las thresholds
            res[i] = True
            
            #this array select the the threshold with the maxiumum TPFP 
            #e.g.
        
            #TPacum = [0,0,1,1,1,2,2]
            #thresh = [0,0,1,1,1,1,2]
            #res    = [0,1,0,0,0,1,1] (1:true, 0:false)
            #res must have []
            
            self.res = res[::-1]
            
            #new modif
    
            #Detecting discontinuities (if the threshold[t] and threshold[t+1] are the same)
            self.idiscontinuous = np.append(np.append(False, ((self.thresholds[:-1] - self.thresholds[1:]) == 0)[1:]), False)
            self.nmeans=nmeans=min(min(int(n/5),8000),max(int(n/50),1000))
            self.isDiscontinuous=~(np.sum(~self.idiscontinuous)>nmeans)
            
            # self.thresholds=self.thresholds-self.minr-1
            self.thresholds[0]+=1e-8
            self.h=1
            
            #Area under the curve
            self.area=self.areaComputation()
            
            #Attribute of the type/kind of TOC curve: TOC, normalized, density, vector,nvector (for normalized vector)
            self.kind='TOC'
            
            #Proportion of positives and data (positive class proportion)
            self.PDataProp=self.npos/self.ntppfp
            
            
            #-->
            #30 equispaced ranks from the first to last rank
            TS = np.linspace(rank[indices[0]], rank[indices[-1]], 30)

            self.unique_ranks = rank[indices][~self.idiscontinuous[1:]]
            
            self.unique_rank_positions = self.TPplusFP[1:][~self.idiscontinuous[1:]]

            P = list([])
            for t in TS:
                ix = np.argmin((self.unique_ranks - t)**2)
                P.append(self.unique_rank_positions[ix])

            self.P = np.array(P)
            self.TS = np.around(TS, 1)  
            
            #self.P, self.TS = self.tickPositions(rank[indices])
            #<--

    def plotHistograms(self):

        """

        This function plots an Histogram of frequencies from the approximated derivative of the TOC.

        """
        fig, ax = plt.subplots(1, 2, figsize = (4.8*2 , 4.8))
     
        X = self.TPplusFP[~self.idiscontinuous]
        Y = self.TP[~self.idiscontinuous]
        M = self.generateMassFunction_secant(X, Y, h = 5000*2)
        

        marker='-r'  
        ax[1].plot(self.thresholds[~self.idiscontinuous], np.array(M)[~self.idiscontinuous], marker,markersize = 1.0, color = "tab:red", label = "Frequency of positive labels")
        ax[1].axhline(y =  Y[-1]/X[-1], color='b', linestyle = '-.', label="Uniform Distribution")
        ax[1].set_xlabel("Ranks")
        ax[1].set_ylabel("Frequency")
        ax[1].set_title("Frequency of positive labels \n in terms of the rank")
        ax[1].legend()


        ax[0].plot(np.arange(self.ndata)[~self.idiscontinuous], np.array(M)[~self.idiscontinuous], marker,markersize = 1.0, color = "tab:red", label = "Frequency of positive labels")
        ax[0].axhline(y =  Y[-1]/X[-1], color='b', linestyle = "-.", label = "Uniform distribution")
        ax[0].set_xlabel("True Positives + False Negatives")
        ax[0].set_ylabel("Frequency")
        ax[0].set_title("Frequency of positive labels \n in terms of the True Positives + False Positives")
        ax[0].legend()

    def plotAttractiveness(self):
        
        """
        
        This function plots an approximated version of the second derivative of the TOC. The second derivative of the toc gives information regarding the appealingness of certain value of the rank associated to positive data.

        """


        fig, ax = plt.subplots(1, 2, figsize = (4.8*2 , 4.8))
        X = self.TPplusFP[~self.idiscontinuous]
        Y = self.TP[~self.idiscontinuous]
        Y = self.generateMassFunction_secant(X, Y, h = 5000*2)
        M = self.generateMassFunction_secant(np.arange(len(Y)), Y, h = 5000*2*2)
        
        ax[0].set_title("Attractiveness")
        ax[0].plot(np.arange(self.ndata)[~self.idiscontinuous] ,np.array(M)[~self.idiscontinuous], label = "Derivative of the Frequency of positive labels", color = "tab:red")
        ax[0].axhline(y = 0, color = 'b', linestyle = "-.", label = "No preference")
        ax[0].set_xlabel("True Positives + False Positives")
        ax[0].legend()

        ax[1].set_title("Attractiveness")
        ax[1].plot(self.thresholds[~self.idiscontinuous] ,np.array(M)[~self.idiscontinuous], label = "Derivative of the Frequency of positive labels", color = "tab:red")
        ax[1].axhline(y = 0, color='b', linestyle = "-.", label = "No preference")
        ax[1].set_xlabel("Ranks")
        ax[1].legend()

    def tickPositions(self, sorted_ranks, n = 30):
        """
        
        This function computes the tick positions of the secondary axes regarding the inherent cummulative distribution of the TOC.
        
        :param sorted_ranks: Received ranks sorted in increase order
    
        :param n:  number of ticks 
    
        :return: Equispaced ticks, position of the equispaced ticks
    
        :rtype: np.array

        """
        TS = np.linspace(sorted_ranks[0], sorted_ranks[-1], 30)
        unique_ranks = sorted_ranks[~self.idiscontinuous[1:]]
        unique_rank_positions = self.TPplusFP[1:][~self.idiscontinuous[1:]]
        P = list([])
        for t in TS:
            ix = np.argmin((self.unique_ranks - t)**2)
            P.append(self.unique_rank_positions[ix])

        return np.array(P), np.around(TS, 1)
    
    def generateMassFunction_secant(self, X, Y, h = 10000):

        """
        This function generates an approximation of the Histogram of frequencies from the TOC. 

        :param X: The Hits + True Positives. They are basically an array from the range 0 to len(Y)

        :param Y: The associated Hits to each Hits + true positive values generated at each Threshold

        :param h: The distance between TOC points to calculate the slope of the secant line.
        
        :return: The approximation to the first derivative of the TOC using the secant method that represents the histogram of frequencies in terms of Hits + False Alarms

        :rtype: np.array



        """

        #Xinterp comienza en 0 y termina con el último valor
        Xinterp = np.arange(0, X[-1]+1)
        Yinterp = np.interp(Xinterp, X, Y)
        M = list([])

        for x in Xinterp[1:]:
            D = (Yinterp[x]-Yinterp[x-h])/h

            if x-h < 0:
                D = (Yinterp[x]-Yinterp[0])/x

            M.append(D)
            
        M  = [M[0]] + M
        
        return M
    

        
    
    
    
    
    
    

    def areaComputation(self):
        """

        This method computes the area under the curve of the TOC and parallelogram and the proportional ratio 
        :return: Returns the TOC's area under the curve
        :rtype: ``float``

        """            
        area=0
        AUC=0
        pararea=0
        if (self.isDiscontinuous):
            h=self.h
            AUC=np.sum(self.TP[~self.idiscontinuous]*h)
            #Area before  TPplusFP_i<npos
            idx=self.TPplusFP[~self.idiscontinuous]<self.npos
            parareaBP=np.sum(h*self.TPplusFP[~self.idiscontinuous][idx])
            idx=self.TPplusFP[~self.idiscontinuous]>=self.npos
            parareaP=h*self.npos*np.sum(idx)
            idx=self.TPplusFP[~self.idiscontinuous]>(self.ntppfp-self.npos)
            xoffset=self.ntppfp-self.npos
            parareaAP=np.sum(h*(self.TPplusFP[~self.idiscontinuous][idx]-xoffset))
            area=AUC-parareaAP
            pararea=parareaBP+parareaP-parareaAP
            self.areaRatio=area/pararea          
        else:
            AUC=np.sum(0.5*(self.TP[:-1]+self.TP[1:])*(self.TPplusFP[1:]-self.TPplusFP[:-1]))
            pararea=self.npos*self.ntppfp
            area=AUC-(self.npos*self.npos)/2

        self.areaRatio=area/pararea
        self.pararea=pararea                
        return AUC

        TOC.areaComputation=areaComputation

    #This method normalize/scales the TOC into the range of [0,1] for both axis
    
    
    def normalize(self):
        """

        This method scales the axis to the interval [0,1]. The self TOC (the TOC which the method is called from) is normalized, there is not new memory allocation.
        :return: Returns the modified TOC curve
        :rtype: ``TOC``

        The ``kind`` TOC curve is *'normalized'*.
        The  true positives plus false positives count is 1, ntppfp=1,
        and true positives, TP=1.
        Nevertheless the basentppfp and basenpos stores the values of the self TOC.

        """    
        self.TPplusFP=self.TPplusFP/self.ntppfp
        self.TP=self.TP/self.ntppfp
        self.h=self.h/self.ntppfp
        self.npos=self.npos/self.ntppfp    
        self.ntppfp=1
        #Area under the curve
        self.area=self.areaComputation()
        self.kind='Normalized'
        return self


    #This method computes the difference between two TOC curves, the 
    def TOCdiff(self,T2):
        """

        This method computes the difference between the self TOC  (the TOC which the method is called from) and the T2 TOC (self-T2). Possibly the best usage with two normlized TOCs, nevertheless the method does not validate such a case. 
        The TOC with the lowest number of data is resampled, to get to TOC with the same number. 

        :param T2: the second TOC that is substracted from the TOC the method is called from (self-T2).

        :ivar kind: *'Diff'*, the resulting TOC is kind 'Diff'.

        :ivar ndata: The number of data in the resulting curve is the greater of the two input TOCs.

        :return: Returns the a new curve with the difference., new memory is allocated.

        :rtype: ``TOC``    

        """

        n1=self.ndata
        n2=T2.ndata
        if (n1<n2):
            Tw1=self.resample(T2)
            Tw2=T2
        if (n2<n1):
            Tw2=T2.resample(self)
            Tw1=self
        Tdiff==copy.deepcopy(Tw1)
        Tdiff.TP=Tw1.TP-Tw2.TP
        Tdiff.npos=max(abs(Tdiff.TP))
        Tdiff.area=(sum(abs(Tdiff.TP))-0.5*abs(Tdiff.TP[-1]))*Tdiff.ntppfp/(Tdiff.ndata-1)
        Tdiff.areaRatio=sum(abs(Tdiff.TP))
        Tdiff.kind='Diff'

        return(Tdiff)

    # Computes a vector representation of the curve using the mean of sets of points
    # the points are those in intervals with equal number of points, if nmeans=30, then the number of points in the 
    # original TOC is divided by 30. 
    def vector(self, nmeans=-1):
        """
        Computes a vector representation of the curve using the mean of sets of points.  
        The points are those in a set of intervals with equal number of points. if ``nmeans=30``, then the number of points in the 
        original TOC is divided by 30. If nmeans=-1 (default) the number of intervals is automatically computed, the maximum possible are 8000 the minimum are 1000 or n/5 (the lower).

        :param nmeans: The number of intervals. Optional parameter that can be automatically computed. Optional

        :param kind: If the input is a normalized TOC the output is a *'nvector'*, otherwise it is a *'verctor'* kind.

        :param npos, ntppfp, basenpos,basentppfp: The counts are taken from the input TOC.

        :return: A TOC instance of the vector representation, new memory is allocated. 

        :rtype: ``TOC``

        """
        n=self.ndata
        if(nmeans==-1):
            nmeans=min(min(int(n/5),8000),max(int(n/50),1000))
        if (nmeans<1 or nmeans>(n/2)):
            print('ERROR: The number of means has to be less than half the number of data in the TOC curve.')
        TV=TOC() 
        TV.kind='Vector'
        TV.h=self.h
        #Data size, this is the total number of true positives plust false positives in the orginal TOC, notice that it is different to ndata
        TV.ntppfp=self.ntppfp
        #This is the number of class 1 in the input data    
        TV.npos=self.npos
        TV.basentppfp=self.basentppfp
        TV.basenpos=self.basenpos
        TV.PDataProp=self.PDataProp       
        if (self.kind=='Normalized'):
            TV.kind='NVector'    

        if self.isDiscontinuous==False or sum(~self.idiscontinuous)>=nmeans:

            ns=nmeans
            y=np.zeros(ns+2)
            x=np.zeros(ns+2)
            xt=np.zeros(ns+2)
            y[0]=0
            x[0]=0
            xt[0]=0
            area=0.0
            for i in range(ns):
                ini=int(i*n/ns)
                ifi=int((i+1)*n/ns)
                xk=(np.linspace(ini,ifi-1,ifi-ini)).astype('int64')
                y[i+1]=np.mean(self.TP[xk])
                x[i+1]=np.mean(self.TPplusFP[xk])
                xt[i+1]=ifi-1

            xt[ns+1]=n-1    
            y[-1]=self.npos
            x[-1]=self.ntppfp
            #Number of segments of the vectorial representation
            TV.ndata=ns+2 
            TV.TPplusFP=x
            TV.TP=y
            TV.thresholds= self.thresholds[xt.astype('int64')]   
            TV.isDiscontinuous=False
            TV.idiscontinuous=np.zeros(ns+2,dtype='bool')
            TV.area=TV.areaComputation()
        else:
            TV.ndata=sum(~self.idiscontinuous)    
            TV.TPplusFP=self.TPplusFP[~self.idiscontinuous]
            TV.TP=self.TP[~self.idiscontinuous]
            TV.thresholds= self.thresholds[~self.idiscontinuous]
            TV.isDiscontinuous=True
            TV.idiscontinuous=np.zeros(TV.ndata,dtype='bool')
            TV.area=TV.areaComputation()                
        return(TV)



        #This method generates a curve with a similar shape than self but with number of samples of T2.
    #The normalized area of the resulting curve approximates that of self
    def resample(self,T2):

        """

        This function creates a new TOC curve,  the data of the self TOC (the TOC which the method is called from) 
        is interpolated to the number of data, and the true positves plus false positves, ``TPplusFP``  in T2.
        Considering that the number of true positives plus false positives is modified (usually extended to a greater value)

        :param T2: the TOC curve with the desired number of data and data values in the TPplusFP axis.

        :return: A TOC of the same kind than T2.

        :rtype: TOC

        """

        n1=self.ndata-1
        n2=T2.ndata-1
        if (n2<n1):
            print("The second curve T2 most have more elements than the first.")


        #Positive proportion    
        pn1=self.basenpos/self.basentppfp
        R=copy.deepcopy(T2)
        #Multiplying to maintain the same proportion of positives and data than self.
        R.basenpos=self.basenpos*R.basentppfp/self.basentppfp
        dfp=R.ntppfp/self.ntppfp
        R.npos=self.npos*dfp
        #The curve R would have fractional values in the TP and TPplusFP
        j=1
        for i in range(n1):
            while(j<=n2  and  (R.TPplusFP[j])<=(self.TPplusFP[i+1])):
                DTPi=(self.TP[i+1]-self.TP[i])*dfp
                DTPpFPi=(self.TPplusFP[i+1]-self.TPplusFP[i])*dfp
                DTPji=R.TPplusFP[j]-self.TPplusFP[i]*dfp
                R.TP[j]=self.TP[i]*dfp+DTPi*DTPji/DTPpFPi   
                j+=1
        R.area=(sum(R.TP)-0.5*R.TP[-1])*R.ntppfp/n2

        if ((R.ntppfp-R.npos)<1e-15):
            R.areaRatio=R.area
        else:
            R.areaRatio=(R.area-R.npoos**2/2)/(R.npos*(R.ntppfp-R.pos))
        return(R)




    #Density function of the TOC and smoothed density
    #Computes the derivative of a TOC, the suggested usage is to input the vector representation of the TOC, 
    #The vector representation of a normalized TOC is a kind of cummulative histogram but instead of break-points we store the middle poiunt of each bin. 
    #Using such vector representation, that can be conceptualized as an empirical cummulative distirbution, we compute the derivative using finite differences,
    #assuming a polynomial function, the finite differences of order 1 are centered differences (second order error term), the finite differences of order 2 are centered differences with fourth order error.
    #Even though the density uses a vector representation, the outputed function could be highly noisy, hence to remnove the noise we apply (if smoothing>0) a mean filter to the density.
    #Additionally to the density and the smoothed density, this function returns an areaError, this is the integral using the trapezoid
    def density(self,smoothing=-1,order=1,verbose=0):

        """

        This function computes an approximation of the derivative of the TOC curve using cubic and fifth order centered finite differences.
        That is to say, if the ``order`` parameter is set to 1, it computes an approximation using a the finite difference before and after the current point, hence
        this approximation provides a cubic error term. Additionally, the function computes a smooth version of the derivative using a mean filter.
        The original TP and TPplusFP arrays from the input TOC (the TOC which this function is called from) are copied.    
        .. warning::
        It is strongly suggested to use a **vector** or **nvector**  kind TOC, because the derivative of a standard TOC only produces finite differences with 0 or 1, hence
        the derivative of a standard TOC is, usually, highly noisy and non-informative. It is strongly suggested to use a **vector** kind TOC produced from a normalized TOC, 
        or a **nvector** TOC, because this kind of TOC have a maximum value of 1 in both access, hence, the derivative can be seen as a density probability function. 
        This is intended use of the function. 

        :param smoothing: window size param for smoothing the derivative. It is -1 by default, so it is computed inside the function trying to present an informative smooth TOC. Optional

        :param order: 1 by defautl. 1 uses formulae for derivatives with a cubic error term, and 2, produces a 5-powered error term. Optional

        :ivar df: density/TOC derivative function approximation (numpy array). This variable can be publicly acccesed in this TOC.

        :ivar smooth: smoothed density/TOC derivative (numpy array). This variable can be publicly acccesed in this TOC..

        :ivar areaSm: area array of the smoothed density (numpy array). The cummulative area of the density, hence it returns for any position the cummulative area from 0 to the postion.

        :ivar areaSm: area array of the non-smoothed density (numpy array). The cummulative area of the density, hence it returns for any position the cummulative area from 0 to the postion.

        :ivar areaError: The area and areaSm are forced to integrate 1, nevertheless the numeric approximation actually do not integrate 1 due to the various approximations that the process uses,
                         This variable stores the actual integral, hence it gives a measure of the error.

        :ivar areaSmError: Similar to areaError but for the smooth version of the TOC.

        :ivar FP: an array of false positives, hence it is possible to plot or to analyze the density vs the FP. 

        :return: A TOC curve of ``kind`` "density", in contrast with other TOC kinds, it includes the variables mentioned above.

        :rtype: ``TOC``

        """

        density = TOC()
        if (self.isDiscontinuous):
            n=np.sum(~self.idiscontinuous)
            #Initializing the density function with zeros
            df=np.zeros(n) 
            #The values of the density function
            y=self.TP[~self.idiscontinuous]
            y=y/np.max(y)
            df[0]=0
            df[1:]=y[1:]-y[:-1]
            density.df=df
            density.h=self.h
            density.sm=df
            density.smwindow=0
            areaS=np.zeros(n)    
            area=np.zeros(n)
            area[:]=np.cumsum(df)
            areaS[:]=area
            density.areaError=np.max(area)
            density.areaSmError=np.max(areaS)
            density.areaUC=area/density.areaError
            density.areaSm=areaS/density.areaSmError
            density.TPplusFP=np.zeros(n)
            density.TP=np.zeros(n)    
            density.TPplusFP[:]=self.TPplusFP[~self.idiscontinuous]
            density.TP=self.TP[~self.idiscontinuous]
            density.thresholds=np.zeros(n)
            density.thresholds[:]=self.thresholds[~self.idiscontinuous]
            density.FP=np.zeros(n)
            density.FP[:]=density.TPplusFP-density.TP
            density.ndata=n
            density.ntppfp=self.ntppfp
            density.npos=self.npos
            density.PDataProp=self.PDataProp
            density.kind='Histogram'
            density.isDiscontinuous=True
            density.idiscontinuous=self.idiscontinuous[~self.idiscontinuous]
        else:
            #We compute the number of data in the input TOC, recall that the number of true positives and false positives are, likely, fractional numbers in the vector representation. 
            n=np.shape(self.TPplusFP)[0]
            #Initializing the density function with zeros
            df=np.zeros(n) 
            #The values of the density function
            y=self.TP
            y=y/np.max(y)
            #The spacing of the x axis in the input TOC, recall that in a standar TOC this value is h=1, becaus the TP+FP axis is increased 1 always, but in the vector representation this is a fractional(rational) value.
            h=self.TPplusFP[1]-self.TPplusFP[0]
            if (self.kind=='Normalized' or self.kind=='NVector'):
                x2=self.TPplusFP*self.ndata-self.TP*self.npos
            else:
                x2=self.TPplusFP-self.TP
            x2=x2/np.max(x2)
            h2=x2[1]-x2[0]
            #Centered finite difference derivatives (error order is 2)
            if (order==1):
                df[1:-1]=(y[2:]-y[0:-2])/(2*h)
                df[0]=df[1]
                df[-1]=df[-2]
            #Centered finite difference derivatives  (higher order, I think this error is order 4)
            if (order==2):
                df[2:-2]=-(y[4:]-y[2:-2])/(12*h)+(y[0:-4]-y[2:-2])/(12*h)+2*(y[3:-1]-y[2:-2])/(3*h)-2*(y[1:-3]-y[2:-2])/(3*h)
                df[0]=df[2]
                df[1]=df[2]
                df[-1]=df[-3]
                df[-2]=df[-3]  
            density.df=df
            density.h=h
            sm=np.zeros(n) 
            #Smoothing the density using a mean filter, it is similar to a uniform kernel with a window size=smoothing
            if (smoothing==-1):
                nw=min(max(int(n/30),50),300)
                smoothing=int(n/nw)
                density.smwindow=smoothing
            if (smoothing>0):
                sm[0:smoothing]= np.mean(df[0:smoothing])
                sm[(n-smoothing):n]=  np.mean(df[(n-smoothing):n])  
                for i in range(smoothing,n-smoothing):
                    sm[i]=np.mean(df[(i-smoothing):(i+smoothing)])  
                density.smooth=sm
            else:
                density.smooth=df #Density without smoothing
                sm[:]=df
            #Area is the integral of the denisty and the smoothing, it is actually a measure of the error. The area must be 1-valued, it is not, because it is computed with the trapezoid method, and the differences are polynomial.
            areaS=np.zeros(n)    
            area=np.zeros(n)
            area[:]=np.concatenate(([0],np.cumsum((df[0:-1]+df[1:])/2)))*h
            areaS[:]=np.concatenate(([0],np.cumsum((sm[0:-1]+sm[1:])/2)))*h
            density.areaError=np.max(area)
            density.areaSmError=np.max(areaS)
            density.areaUC=area/density.areaError
            density.areaSm=areaS/density.areaSmError
            density.TPplusFP=np.zeros(n)
            density.TP=np.zeros(n)    
            density.TPplusFP[:]=self.TPplusFP
            density.TP=self.TP
            density.thresholds=np.zeros(n)
            density.thresholds[:]=self.thresholds
            density.FP=np.zeros(n)
            density.FP[:]=x2
            density.factorFP=h2/h
            density.ndata=self.ndata
            density.npos=self.npos
            density.ntppfp=self.ntppfp
            density.PDataProp=self.PDataProp
            density.kind='Density'

        return(density)


    #This function computes a probability value given 

    def rank2prob(self,rank,kind='Density',indices=None):

        """

        This function computes probability values associated to a rank value. The ``thresholds`` array of the density TOC is used for this purpose. 
        Very possibly the rank is the same than those used to compute a standard TOC instantiated by TOC(rank,groundtruth), hence the indices used 
        in the constructor are available and can save computational time. Otherwise the indices are recomputed. In any case the inputed ``rank``
        array must be in the same interval than the thresholds.

        :param rank: A numpy array with the ran. The intended uses is that this rank comes from the standard TOC computation, and to associate this rank with geolocations, hence the probabilities can be associated in the same order. 

        :param kind: if ``kind`` is ''density'' the probabilitites are computed with the non-smoothed density function, otherwise the smooth version is used. Notice that a this function only must be called from a ''density'' kind TOC. Optional

        :param indices: Indices of the reversely sorted rank, this array is computed by the standard TOC computation. Hence the computational cost of recomputing them could be avoided, otherwise the indices are recomputed and they are not stored. Optional 

        return: a numpy array with the probabilities. The probabilities do not sum 1, instead they sum ``PDataProp``, that is to say they sum the proportion of positives in the data. That is an estimation of the probability of having a 1-class valued datum.

        :rtype: numpy array

        """

        if (indices==None):
            #Sorting the classification rank and getting the indices
            indices=sorted(range(len(rank)),key=lambda index: rank[index],reverse=True)
        if (kind=='Density'):
            df=self.df
        else:
            df=self.smooth
        nr=len(rank)    
        nd=self.ndata-1
        prob=np.zeros(nr)
        j=0
        for i in range(nd):
            while(j<nr  and  (rank[indices[j]])>=(self.thresholds[i+1])):
                prob[indices[j]]=df[i]
                j+=1          
        prob=(prob/np.sum(prob))*self.PDataProp       
        return(prob)








    def __plotTOC(self, filename = '', title='default', TOCname='TOC', kind='TOC', 
                  height=1800, width=1800, dpi=300, xlabel="default", ylabel="default"): #que el usuario puede controlar si quiere ver la versión discontinua o continua de la curva TOC, aunque si ponemos dos plots, uno con los puntos de cada TPFP maximo de cada threshold y otro con el marcador de línea se estarían mostrando ambos casos en una sola grafica. Lo del caso continuo y discreto mas bien aplicaría para el la PMF/Histograma.

        fig, ax1 = plt.subplots(1, 1, figsize = (4.8 , 4.8))

        if (filename!=''):
            fig=plt.figure(figsize=(height/dpi, width/dpi), dpi=dpi)

        if(xlabel=="default"):
            xlabel="True Positives plus False Positives"
            if (kind=='tTOC'):
                xlabel="Tresholds"

        if(ylabel=="default"):
            ylabel="True Positives"  
        marker='-r'    

        #if (self.isDiscontinuous):
        if (True):        
            marker='.r'
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)  

        if (title=='default'):
            if (self.kind=='TOC'):
                title="Total Operating Characteristic"
            if (kind=='tTOC'):
                title="True Positives vs Thresholds"
                TOCname='TP'
    #    plt.title(title)

        if(kind=='Normalized'):        
            rx = np.array([0,self.basenpos/self.basentppfp,1,1-self.basenpos/self.basentppfp])
            ry = np.array([0,self.basenpos/self.basentppfp,self.basenpos/self.basentppfp,0])            
            plt.ylim(0, 1.01*self.basenpos/self.basentppfp)
            plt.xlim(0, 1.01)            
            plt.text(0.575,0.025*self.basenpos/self.basentppfp,'AUC=')
            plt.text(0.675,0.025*self.basenpos/self.basentppfp,str(round(self.areaRatio,4)))
            plt.plot(rx, ry,'b--')
            plt.plot(self.TPplusFP[~self.idiscontinuous]/self.ntppfp,self.TP[~self.idiscontinuous]/self.npos,marker,label=TOCname,linewidth=2)

        elif (kind=='TOC'):


            rx = np.array([0, self.npos, self.ntppfp, self.ntppfp-self.npos, 0])
            ry = np.array([0, self.npos, self.npos, 0, 0])

            ax1.set_ylim(0, 1.01*self.npos)
            ax1.set_xlim(0, 1.01*self.ntppfp)
            ax1.text(0.575*self.ntppfp, 0.025*self.npos, 'AUC = ')
            ax1.text(0.675*self.ntppfp, 0.025*self.npos, str(round(self.areaRatio, 4)))

            ax1.plot(np.array([0, self.ntppfp]), np.array([0, self.npos]),'b-.',
                     label = "Uniform Distribution")

            ax1.plot(rx, ry, '--')

            ax1.plot(self.TPplusFP[~self.idiscontinuous], 
                     self.TP[~self.idiscontinuous],
                     marker,markersize = 5.0,
                     label = TOCname, linewidth = 1)

            ax1.plot(self.TPplusFP[~self.idiscontinuous], 
                     self.TP[~self.idiscontinuous], 
                     "--", markersize = 0.5, 
                     label = "Slope", 
                     linewidth = 1)
            
            X = self.TPplusFP[~self.idiscontinuous]

            Y = self.TP[~self.idiscontinuous]

           # M = self.generateMassFunction_secant(X, Y, h = 5000)
         
            #ax1.plot(np.array(M)*self.npos, 
            #         marker, markersize = 0.01,
            #         label = "Histogram", linewidth = 0.001, color = "tab:gray")
            
            #ax1.axhline(y = self.npos/len(self.TP)*self.npos, color = "blue", linestyle = "-.")
            

            ax1.legend(loc = 'upper left')


            #Preparing the overlaped plot with the top and right axis for the CPF
            ax2 = fig.add_subplot(111, label = "2", frame_on = False)

            #posiciones de los ticks chiquitos

            unique_rank_positions = self.TPplusFP[1:][~self.idiscontinuous[1:]]

            P, Q = self.tickPositions(sorted_ranks = self.rank[self.indices])

            ax2.set_xticks(P)
            ax2.xaxis.set_minor_locator(ticker.FixedLocator(P))

            ax2.set_xlabel('X', color = "tab:blue")  
            ax2.set_ylabel("P($x \leq X$)", color = "tab:blue")  
            ax2.tick_params(labeltop = True)   
            ax2.tick_params(labelright = True)
            ax2.tick_params(labelbottom = False)   
            ax2.tick_params(labelleft = False)
            ax2.xaxis.tick_top()

            #los ticks grandes

            n = np.argmax(self.P[1:]-self.P[:-1])
            ax2.xaxis.set_major_locator(ticker.FixedLocator(np.unique(np.array(self.P)[[0, n, n+1, -1]])))

            #las etiquetas de los ticks grandes
            ax2.set_xticklabels(np.unique(np.array(self.TS)[[0, n, n+1, -1]]), rotation = 90)

            ax2.yaxis.tick_right()
            ax2.yaxis.set_minor_locator(AutoMinorLocator())
            ax2.xaxis.set_label_position('top')
            ax2.yaxis.set_label_position('right')
            ax2.set_title(title)


        elif (kind=='tTOC'):
            plt.ylim(0, 1.01*self.npos)
            plt.xlim(self.thresholds[0]*0.98, 1.01*self.thresholds[-1])
            plt.plot(self.thresholds[~self.idiscontinuous],self.TP[~self.idiscontinuous],marker,label=TOCname,linewidth=1)   



        if (filename!=''):
            plt.savefig(filename,dpi=dpi)
        else:
            plt.show(block=True)
        plt.close("all")
        plt.clf()
        gc.collect() 



    def __plotVector(self, filename = '',title='default',TOCname='TOC',kind='Vector',height=1800,width=1800,dpi=300,xlabel="default",ylabel="default"):
        if (filename!=''):
            fig=plt.figure(figsize=(height/dpi, width/dpi), dpi=dpi)
        if(xlabel=="default"):
            xlabel="True Positives plus False Positives"
            if (kind=='tVector'):
                xlabel="Tresholds"
        if(ylabel=="default"):
            ylabel="True Positives"              
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)  
        if (title=='default'):
            title="Approximation of TOC via average vectors"
            if (kind=='tVector'):
                title="True Positives vs Thresholds"
                TOCname='TP'
        plt.title(title)
        marker='r-'
        if (self.isDiscontinuous):
            marker='.r'     
        if (kind=='Vector'):    
            rx=np.array([0,self.npos,self.ntppfp,self.ntppfp-self.npos])
            ry=np.array([0,self.npos,self.npos,0])
            plt.ylim(0, 1.01*self.npos)
            plt.xlim(0, 1.01*self.ntppfp)
            plt.text(0.575*self.ntppfp,0.025*self.npos,'AUC=')
            plt.text(0.675*self.ntppfp,0.025*self.npos,str(round(self.areaRatio,4)))
            plt.plot(rx, ry,'b--')
            plt.plot(self.TPplusFP,self.TP,marker,label=TOCname)
        elif (kind=='tVector'):
            plt.ylim(0, 1.01*self.npos)
            plt.xlim(self.thresholds[0]*0.98, 1.01*self.thresholds[-1])
            plt.plot(self.thresholds,self.TP,marker,label=TOCname)

        if (filename!=''):
            plt.savefig(filename)
        else:
            plt.show(block=True)
        plt.close("all")
        plt.clf()
        gc.collect() 
    


    def __plotDiff(self,filename = '',title='default',TOCname='TOC',kind='Diff',height=1800,width=1800,dpi=300,xlabel="default",ylabel="default"):
        if (filename!=''):
            fig=plt.figure(figsize=(height/dpi, width/dpi), dpi=dpi)
        if(xlabel=="default"):
            xlabel="True Positives plus False Positives"
        if(ylabel=="default"):
            ylabel="True Positives"              
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)  
        if (title=='default'):
            title="Difference between 2 TOCs"
        marker='r-'
        if (self.isDiscontinuous):
            marker='.r'        
        plt.title(title)
        plt.ylim(-1.01, 1.01)
        plt.xlim(-0.01, 1.01)            
        plt.text(0.575,0.025,'AUC=')
        plt.text(0.675,0.025,str(round(self.areaRatio,4)))
        plt.plot(self.TPplusFP/self.ntppfp,self.TP/self.npos,marker,label=TOCname,linewidth=3)
        plt.legend(loc='lower right')   
        if (filename!=''):
            plt.savefig(filename,dpi=dpi)
        else:
            plt.show(block=True)
        plt.close("all")
        plt.clf()
        gc.collect() 
    



    def __plotNvector(self,filename = '',title='default',TOCname='TOC',kind='nvector',height=1800,width=1800,dpi=300,xlabel="default",ylabel="default"):
        if (filename!=''):
            fig=plt.figure(figsize=(height/dpi, width/dpi), dpi=dpi)
        if(xlabel=="default"):
            xlabel="True Positives plus False Positives"
        if(ylabel=="default"):
            ylabel="True Positives"               
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)  
        if (title=='default'):
            title="Approximation of normalized TOC via average vectors"
        marker='r-'
        if (self.isDiscontinuous):
            marker='.r'         
        plt.title(title)
        rx=np.array([0,self.npos,self.ntppfp,self.ntppfp-self.npos])
        ry=np.array([0,self.npos,self.npos,0])
        plt.ylim(0, 1.01*self.npos)
        plt.xlim(0, 1.01*self.ntppfp)
        plt.text(0.575*self.ntppfp,0.025*self.npos,'AUC=')
        plt.text(0.675*self.ntppfp,0.025*self.npos,str(round(self.areaRatio,4)))
        plt.plot(rx, ry,'b--')
        plt.plot(self.TPplusFP,self.TP,marker,label=TOCname) 
        if (filename!=''):
            plt.savefig(filename,dpi=dpi)
        else:
            plt.show(block=True)
        plt.close("all")
        plt.clf()
        gc.collect() 
    
    

    def __plotNormalized(self,filename = '',title='default',TOCname='TOC',kind='normalized',height=1800,width=1800,dpi=300,xlabel="default",ylabel="default"):
        if (filename!=''):
            fig=plt.figure(figsize=(height/dpi, width/dpi), dpi=dpi)
        if (title=='default'):
            title="Normalized Total Operating Characteristic"
        if(xlabel=="default"):
            xlabel="True Positives plus False Positives"
        if(ylabel=="default"):
            ylabel="True Positives"   
        marker='r-'
        if (self.isDiscontinuous):
            marker='.r'         
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)          
        plt.title(title)
        rx=np.array([0,self.npos,self.ntppfp,self.ntppfp-self.npos])
        ry=np.array([0,self.npos,self.npos,0])            
        plt.ylim(0, 1.01*self.npos)
        plt.xlim(0, 1.01*self.ntppfp)         
        plt.text(0.575*self.ntppfp,0.025*self.npos,'AUC=')
        plt.text(0.675*self.ntppfp,0.025*self.npos,str(round(self.areaRatio,4)))
        plt.plot(rx, ry,'b--')
        plt.plot(self.TPplusFP[~self.idiscontinuous],self.TP[~self.idiscontinuous],marker,label=TOCname)
        plt.legend(loc='lower right')
        if (filename!=''):
            plt.savefig(filename,dpi=dpi)
        else:
            plt.show(block=True)
        plt.close("all")
        plt.clf()
        gc.collect() 
    

    def __plotDensity(self,filename = '',title='default',TOCname='TOC',kind='density',height=1800,width=1800,dpi=300,xlabel="TP+FP",ylabel="Density"):
        if (title=='default'):
            if (kind=='Smoothed'):
                title="Smoothed vs TP+FP, quartiles"
            elif(kind=='Density'):
                title="Density vs TP+FP, quartiles"
            elif(kind=='Histogram'):
                title="Histogram of discontinuos probability and quartiles"
            elif(kind=='tHistogram'):
                title="Histogram of discontinuos probability vs Thresholds and quartiles"            
            elif(kind=='tDensity'):
                title="Probability versus Thresholds"            
            elif(kind=='tSmoothed'):
                title="Smoothed probability versus Thresholds"                        
        if (xlabel=="default"):        
            xlabel="True Positives plus False Positives"
            if(kind=='tDensity' or  kind=='tSmoothed' or kind=='tHistogram'):
                xlabel="Thresholds"            

        if (ylabel=="default" and (kind=='Density' or kind=='Smoothed' or kind=='tDensity' or kind=='tSmoothed')):
            ylabel="Density"
        if (ylabel=="default" and (kind=='Histogram' or kind=='tHistogram')):
            ylabel="Probability"
        marker='r-'

        if (self.isDiscontinuous):
            marker='.r'    

        plt.title(title)    
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        i1=np.argmax(self.areaUC>0.25)
        i2=np.argmax(self.areaUC>0.5)
        i3=np.argmax(self.areaUC>0.75)
        if (kind=='Smoothed'):
            plt.fill_between(self.TPplusFP[0:i1],self.smooth[0:i1],color='#f54242')
            plt.fill_between(self.TPplusFP[i1:i2],self.smooth[i1:i2],color='#f57842')
            plt.fill_between(self.TPplusFP[i2:i3],self.smooth[i2:i3],color='#f5e042')
            plt.fill_between(self.TPplusFP[i3:],self.smooth[i3:],color='#aaf542')
            plt.plot(self.TPplusFP, self.smooth,marker)
        elif (kind=='Density'):
            plt.fill_between(self.TPplusFP[0:i1],self.df[0:i1],color='#6042f5')
            plt.fill_between(self.TPplusFP[i1:i2],self.df[i1:i2],color='#c242f5')
            plt.fill_between(self.TPplusFP[i2:i3],self.df[i2:i3],color='#f54290')
            plt.fill_between(self.TPplusFP[i3:],self.df[i3:],color='#f5424b')
            plt.plot(self.TPplusFP, self.df,marker)
        elif (kind=='tDensity'):
            plt.fill_between(self.thresholds[0:i1],self.df[0:i1],color='#6042f5')
            plt.fill_between(self.thresholds[i1:i2],self.df[i1:i2],color='#c242f5')
            plt.fill_between(self.thresholds[i2:i3],self.df[i2:i3],color='#f54290')
            plt.fill_between(self.thresholds[i3:],self.df[i3:],color='#f5424b')
            plt.plot(self.thresholds, self.df,marker)
        elif (kind=='Histogram'):
            plt.bar(self.TPplusFP[0:i1],self.df[0:i1],color='#0390fc',width=self.ntppfp/100)
            plt.bar(self.TPplusFP[i1:i2],self.df[i1:i2],color='#78d902',width=self.ntppfp/100)
            plt.bar(self.TPplusFP[i2:i3],self.df[i2:i3],color='#f0c000',width=self.ntppfp/100)
            plt.bar(self.TPplusFP[i3:],self.df[i3:],color='#ae2029',width=self.ntppfp/100)
            plt.plot(self.TPplusFP, self.df,marker)
        elif (kind=='tHistogram'):
            tw=np.max(self.thresholds)-np.min(self.thresholds)
            plt.bar(self.thresholds[0:i1],self.df[0:i1],color='#0390fc',width=tw/200)
            plt.bar(self.thresholds[i1:i2],self.df[i1:i2],color='#78d902',width=tw/200)
            plt.bar(self.thresholds[i2:i3],self.df[i2:i3],color='#f0c000',width=tw/200)
            plt.bar(self.thresholds[i3:],self.df[i3:],color='#ae2029',width=tw/200)
            plt.plot(self.thresholds, self.df,marker)        
        elif (kind=='tSmoothed'):
            plt.fill_between(self.thresholds[0:i1],self.smooth[0:i1],color='#f54242')
            plt.fill_between(self.thresholds[i1:i2],self.smooth[i1:i2],color='#f57842')
            plt.fill_between(self.thresholds[i2:i3],self.smooth[i2:i3],color='#f5e042')
            plt.fill_between(self.thresholds[i3:],self.smooth[i3:],color='#aaf542')
            plt.plot(self.thresholds, self.smooth,marker)
        if (filename!=''):
            plt.savefig(filename,dpi=dpi)
        else:
            plt.show(block=True)
        plt.close("all")
        plt.clf()
        gc.collect() 
    



        #Private function assignation      
        TOC.__plotTOC=__plotTOC
        TOC.__plotDiff=__plotDiff
        TOC.__plotVector=__plotVector
        TOC.__plotNvector=__plotNvector
        TOC.__plotNormalized=__plotNormalized
        TOC.__plotDensity=__plotDensity

    #This function plots the TOC to the terminal or to a file
    def plot(self, filename = '',title='default',TOCname='TOC',kind='None',height=2400,width=2400,dpi=300,xlabel="default",ylabel="default"):

        """
        A generic plot function for all the kind of TOCs.  All the parameters are optional. If ``filename`` is not given it plots to a window, otherwise it is a png file.

        :param filename: Optional. If given it must be a png filename, otherwise the TOC is plotted to a window.

        :param title: Optional, title of the plot.

        :param kind: Optional, a standard TOC can be plotted normalized or in the original axis values.

        :param height: pixels of the height. 1800 by default.

        :param width: pixels of the width. 1800 by default.

        :param dpi: resolution. 300 by default.

        :param xlabel: string.

        :param ylabel: string.

        :return: it does not return anything.

        """

        print(kind)

        if (kind=='None'):
            kind=self.kind

        if (kind=='TOC' or kind=='tTOC'):
            self.__plotTOC(filename, title, TOCname, kind, height, width, dpi, xlabel, ylabel)

        if(kind=='Diff'):
            self.__plotTOC(filename,title,TOCname,kind,height,width,dpi,xlabel,ylabel)

        if(kind=='Vector' or kind=='tVector'):
            self.__plotVector(filename,title,TOCname,kind,height,width,dpi,xlabel,ylabel)
        if(kind=='NVector'):
            self.__plotNvector(filename,title,TOCname,kind,height,width,dpi,xlabel,ylabel)
        if (kind=='Normalized'):
            self.__plotNormalized(filename,title,TOCname,kind,height,width,dpi,xlabel,ylabel)
        if(kind=='Density' or kind=='Histogram' or kind=='Smoothed' or kind=='tSmoothed' or kind=='tDensity' or kind=='tHistogram'):
            self.__plotDensity(filename,title,TOCname,kind,height,width,dpi,xlabel,ylabel)



            return M

            TOC.normalize = normalize
            TOC.vector = vector
            TOC.resample = resample
            TOC.diff = TOCdiff
            TOC.density = density
            TOC.rank2prob = rank2prob
            TOC.plot = plot

                        #Private function assignation      
            TOC.__plotTOC=__plotTOC
            TOC.__plotDiff=__plotDiff
            TOC.__plotVector=__plotVector
            TOC.__plotNvector=__plotNvector
            TOC.__plotNormalized=__plotNormalized
            TOC.__plotDensity=__plotDensity
