'''
Created on Jan 10, 2016

@author: Mick Hart
'''
import timeit
import sys
import pandas as pd
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import somcuda
import pycuda.autoinit
import pycuda.driver as pycuda
from collections import defaultdict

import random

MAX_THREADS_PER_BLOCK = 512
MAX_BLOCKS_PER_GRID = 512
INF = np.float32(pow(2,32)/2-1)
MAX_FEAT = INF/4.0
COLOR_MAP = False #allow this code to run the benchmark, three-color SOM

class Process:
    EvaluateTrainingData, ColorSOMDemo, TrainSOM  = range(3)
class Metric:
    Euclidean, AbsoluteValue = range(2)

def postAnalyze(somMatrix, featuresForSOM):
    print "Performing post-train analysis..."
    somMatrixPtr = pycuda.mem_alloc(somMatrix.nbytes)
    somBytesPerRow = np.int32(somMatrix.strides[0])
    somNumberOfRows = np.int32(somMatrix.shape[0])
    somNumberOfColumns = np.int32(somMatrix.shape[1])
    pycuda.memcpy_htod(somMatrixPtr,somMatrix)
    #allocate space for bmu index
    bmu = np.zeros(somMatrixRows).astype(np.float32)
    bmuPtr = pycuda.mem_alloc(bmu.nbytes)
    pycuda.memcpy_htod(bmuPtr,bmu)
    bmuIndex = np.zeros(somMatrixRows).astype(np.int32)
    bmuIndexPtr = pycuda.mem_alloc(bmuIndex.nbytes)
    pycuda.memcpy_htod(bmuIndexPtr,bmuIndex)
    hitCountDict = defaultdict(list)
    for i in range (0,len(featuresForSOM)):
        feats = featuresForSOM.loc[i].as_matrix().astype(np.float32)
        featuresPtr = pycuda.mem_alloc(feats.nbytes)
        pycuda.memcpy_htod(featuresPtr,feats)
        #find the BMU
        computeBMU(somMatrixPtr, bmuPtr, bmuIndexPtr, featuresPtr, np.int32(len(featuresForSOM.columns)),  somBytesPerRow, somNumberOfRows, somNumberOfColumns, np.float32(MAX_FEAT), np.float32(INF),np.int32(metric),block=(blk,1,1),grid=(somNumberOfRows,1))
        pycuda.memcpy_dtoh(bmu,bmuPtr)
        pycuda.memcpy_dtoh(bmuIndex,bmuIndexPtr)
        block = np.argmin(bmu)
        thread = bmuIndex[block]
        val = hitCountDict[(block,thread)]
        if val == None or len(val) == 0:
            hitCountDict[(block,thread)] = [1,i]
        else:
            hitCountDict[(block,thread)][0] += 1

    keys = hitCountDict.keys()
    for i in keys:
        if hitCountDict[i][0] == 1:
            rowIdx = hitCountDict[i][1]
            block = i[0]
            thread = i[1]
            print "block: %i thread: %i single hit on observation: %i" % (i[0],i[1],hitCountDict[i][1])
            #for j in range (len(featuresForSOM)-62,len(featuresForSOM)-1):
            #   print featuresForSOM.ix[i,featuresForSOM.columns[j]]
            obs = somMatrix[block][thread]
            for i in range (len(featuresForSOM.columns)-62,len(featuresForSOM.columns)):
                obs[i] = featuresForSOM.ix[rowIdx,featuresForSOM.columns[i]]
            #obs = [featuresForSOM.ix[rowIdx,featuresForSOM.columns[j]] for j in range (len(featuresForSOM.columns)-62,len(featuresForSOM.columns)) ] #retain original predictions
            somMatrix[block][thread] = obs # replace with the original observation
            
def confirmInitialization(featuresForSOM,somMatrix):
    #allocate memory for the somcuda on the device
    somMatrixPtr = pycuda.mem_alloc(somMatrix.nbytes)
    somBytesPerRow = np.int32(somMatrix.strides[0])
    somNumberOfRows = np.int32(somMatrix.shape[0])
    somNumberOfColumns = np.int32(somMatrix.shape[1])
    pycuda.memcpy_htod(somMatrixPtr,somMatrix)
    #allocate space for bmu index
    bmu = np.zeros(somMatrixRows).astype(np.float32)
    bmuPtr = pycuda.mem_alloc(bmu.nbytes)
    pycuda.memcpy_htod(bmuPtr,bmu)
    bmuIndex = np.zeros(somMatrixRows).astype(np.int32)
    bmuIndexPtr = pycuda.mem_alloc(bmuIndex.nbytes)
    pycuda.memcpy_htod(bmuIndexPtr,bmuIndex)
    intraDayOffset = features.columns.get_loc('Ret_121')
    dayOffset = features.columns.get_loc('Ret_PlusOne')
    objVal = 0.0;
    objSampSize=0.0
    r = [[[0.0 for k in range(0,3)] for i in range(somMatrixColumns)] for j in range (somMatrixRows)] 
    nodeHitMatrix = np.array(r).astype(np.float32)
    hitCountDict = defaultdict(list)
    samples = [x for x in range (0, somMatrixRows*somMatrixColumns)]
    if len(samples) >= len(featuresForSOM):
        samples = [x for x in range (0, len(featuresForSOM))]       
    for i in samples:
        feats = featuresForSOM.loc[i].as_matrix().astype(np.float32)
        featuresPtr = pycuda.mem_alloc(feats.nbytes)
        pycuda.memcpy_htod(featuresPtr,feats)
        #find the BMU
        computeBMU(somMatrixPtr, bmuPtr, bmuIndexPtr, featuresPtr, np.int32(len(featuresForSOM.columns)),  somBytesPerRow, somNumberOfRows, somNumberOfColumns, np.float32(MAX_FEAT), np.float32(INF),np.int32(metric),block=(blk,1,1),grid=(somNumberOfRows,1))
        pycuda.memcpy_dtoh(bmu,bmuPtr)
        pycuda.memcpy_dtoh(bmuIndex,bmuIndexPtr)
        block = np.argmin(bmu)
        thread = bmuIndex[block]
        val = hitCountDict[(block,thread)]
        if val == None or len(val) == 0:
            hitCountDict[(block,thread)] = [1,i]
        else:
            hitCountDict[(block,thread)][0] += 1
        val = np.int32(hitCountDict[(block,thread)])[0]
        if val == 1:
            val = 0x0000ff00
        elif val <= 10:
            val = 0x000000ff
        elif val <= 100:
            val = 0x00ff0000
        else:
            val = 0x00ffffff
        bval = (val & 0x000000ff)
        gval = ((val & 0x0000ff00) >> 8)
        rval = ((val & 0x00ff0000) >> 16)
        nodeHitMatrix[block][thread] = [rval/255.0,gval/255.0,bval/255.0]
    fig20 = plt.figure(20,figsize=(6*3.13,4*3.13))
    fig20.suptitle('Train Node Hit Counts. Black: 0 Green: 1 Blue: <=10 Red: <=100 White >100', fontsize=20)
    ax = plt.subplot(111)
    somplot = plt.imshow(nodeHitMatrix,interpolation="none")
    plt.show()
    plt.pause(0.1)

def markFeatures(featuresForSOM,retainFeatures):
    featureStats = pd.DataFrame(index=['Mean','Std','NormedVar'],columns=featuresForSOM.columns)
    for i in range(0,len(featuresForSOM.columns)):
        mean = featuresForSOM.ix[:,i].mean()
        std = featuresForSOM.ix[:,i].std()
        nMean = 0.0
        nVar = 1.0
        if normalizeMean:
            nMean= mean
        if normalizeVar:
            nVar = std
        featureStats.loc['Mean',featureStats.columns[i]] = nMean
        featureStats.loc['Std',featureStats.columns[i]] = nVar        
        featureStats.loc['NormedVar',featureStats.columns[i]] = featuresForSOM.ix[:,i].var() #extract variance of normed features

    for i in range (0,121):
        if featureStats.columns[i] not in retainFeatures:
            featuresForSOM.loc[:,featureStats.columns[i]] = [2.0*MAX_FEAT for k in range(0,len(featuresForSOM))]

    return featureStats
    
def initializeData():
    print "Initializing data and SOM matrix..."
    #replace NaN in featuresForSOM to MAX_FEAT - these will be ignored in the BMU kernel and neighborhood adjustment kernel
    featuresForSOM.fillna(2.0*MAX_FEAT,inplace=True)
    #dfs for column stats
    featureStats = pd.DataFrame(index=['Mean','Std','NormedVar'],columns=featuresForSOM.columns)
    #color data only - this is used for testing
    if COLOR_MAP:
        for i in range(0,len(featuresForSOM.columns)):
            featuresForSOM.ix[:,i] = [x/255.0 for x in featuresForSOM.ix[:,i]] 
    #normalize the data - saving the stats
    print "-- Normalizing the data..."
    for i in range(0,len(featuresForSOM.columns)):
        mean = featuresForSOM.ix[:,i].mean()
        std = featuresForSOM.ix[:,i].std()
        nMean = 0.0
        nVar = 1.0
        if normalizeMean:
            nMean = mean
        if normalizeVar:
            nVar = std
        featureStats.loc['Mean',featureStats.columns[i]] = nMean
        featureStats.loc['Std',featureStats.columns[i]] = nVar
        featuresForSOM.ix[:,i] = [(x-nMean)/nVar for x in featuresForSOM.ix[:,i]] 
        featureStats.loc['NormedVar',featureStats.columns[i]] = featuresForSOM.ix[:,i].var() #extract variance of normed features
        featureStatsMatrix = np.array(featureStats).astype(np.float32)

    #create the SOM matrix with random initialization
    print"-- Building the %i by %i SOM..." % (somMatrixRows,somMatrixColumns)
    r = [[ [random.random() for k in range(len(featuresForSOM.columns))] for i in range(somMatrixColumns)] for j in range (somMatrixRows)] 
    somMatrix = np.array(r).astype(np.float32)

    somMatrixPtr = pycuda.mem_alloc(somMatrix.nbytes)
    somBytesPerRow = np.int32(somMatrix.strides[0])
    somNumberOfRows = np.int32(somMatrix.shape[0])
    somNumberOfColumns = np.int32(somMatrix.shape[1])
    pycuda.memcpy_htod(somMatrixPtr,somMatrix)
    print "Initialization complete"
    #confirmInitialization(featuresForSOM,somMatrix)
    return featuresForSOM, featureStatsMatrix, somMatrix

def computeAvgDistancetoBMU(currentIter,iterationDistance, features, nodeHitMatrix, somMatrixPtr, somMatrix, featureStatsMatrix, featuresPtr, featureCount, somBytesPerRow, somNumberOfRows, somNumberOfColumns):
    adjustNodes = {}
    sampSize = 0
    cumDistance = 0.0
    nodeHitMatrix.fill(0)
    hitCountDict.clear()
    if len(featuresForSOM) < 100:
        sampSize = len(featuresForSOM)
    elif currentIter < len(featuresForSOM):
        sampSize = int(currentIter)
        if sampSize == 0:
            sampSize = min(somNumberOfRows*somNumberOfColumns,len(featuresForSOM))
    else:
        sampSize = len(featuresForSOM)
    samples = [x for x in range (0,sampSize)]
    #allocate space for bmu
    bmu = np.zeros(somMatrixRows).astype(np.float32)
    bmuPtr = pycuda.mem_alloc(bmu.nbytes)
    pycuda.memcpy_htod(bmuPtr,bmu)
    #allocate space for bmu index
    bmuIndex = np.zeros(somMatrixRows).astype(np.int32)
    bmuIndexPtr = pycuda.mem_alloc(bmuIndex.nbytes)
    pycuda.memcpy_htod(bmuIndexPtr,bmuIndex)
    for i in samples:
        feats = featuresForSOM.loc[i].as_matrix().astype(np.float32)
        featuresPtr = pycuda.mem_alloc(feats.nbytes)
        pycuda.memcpy_htod(featuresPtr,feats)
        #find the BMU
        computeBMU(somMatrixPtr, bmuPtr, bmuIndexPtr, featuresPtr, np.int32(featureCount),  somBytesPerRow, somNumberOfRows, somNumberOfColumns, np.float32(MAX_FEAT), np.float32(INF),np.int32(metric),block=(blk,1,1),grid=(somNumberOfRows,1))
        pycuda.memcpy_dtoh(bmu,bmuPtr)
        pycuda.memcpy_dtoh(bmuIndex,bmuIndexPtr)
        cumDistance += np.min(bmu)
        block = np.argmin(bmu)
        thread = bmuIndex[block]
        adjustNodes[i]=[block,thread]
        val = hitCountDict[(block,thread)]
        if val == None or len(val) == 0:
            hitCountDict[(block,thread)] = [1,i]
        else:
            hitCountDict[(block,thread)][0] += 1
        val = np.int32(hitCountDict[(block,thread)])[0]
        if val == 1:
            val = 0x0000ff00
        elif val <= 10:
            val = 0x000000ff
        elif val <= 100:
            val = 0x00ff0000
        else:
            val = 0x00ffffff
        bval = (val & 0x000000ff)
        gval = ((val & 0x0000ff00) >> 8)
        rval = ((val & 0x00ff0000) >> 16)
        nodeHitMatrix[block][thread] = [rval/255.0,gval/255.0,bval/255.0]
    iterationDistance.append(cumDistance/sampSize)
    iterationCount.append(currentIter)
    return cumDistance/sampSize

def trainSOM(features, somMatrix):
    print "Training the (%i,%i)-dimensional SOM for %f iterations." % (somMatrixRows,somMatrixColumns,maxIterations)    
    print "-- Moving data from host to device..."
    r = [[[0.0 for k in range(0,3)] for i in range(somMatrixColumns)] for j in range (somMatrixRows)] 
    nodeHitMatrix = np.array(r).astype(np.float32)
    hitCountPercent = []
    oneHits = []
    maxHits = []
    distinctHits = []
    start_time = timeit.default_timer()
    featureCount = len(featuresForSOM.columns)
    #allocate memory for the somcuda on the device
    somMatrixPtr = pycuda.mem_alloc(somMatrix.nbytes)
    somBytesPerRow = np.int32(somMatrix.strides[0])
    somNumberOfRows = np.int32(somMatrix.shape[0])
    somNumberOfColumns = np.int32(somMatrix.shape[1])
    pycuda.memcpy_htod(somMatrixPtr,somMatrix)
    #allocate memory for the feature stats on the device
    featureStatsMatrixPtr = pycuda.mem_alloc(featureStatsMatrix.nbytes)
    featureStatsBytesPerRow = np.int32(featureStatsMatrix.strides[0])
    featureStatsNumberOfRows = np.int32(featureStatsMatrix.shape[0])
    pycuda.memcpy_htod(featureStatsMatrixPtr,featureStatsMatrix)
    
    #allocate space for bmu
    bmu = np.zeros(somMatrixRows).astype(np.float32)
    bmuPtr = pycuda.mem_alloc(bmu.nbytes)
    pycuda.memcpy_htod(bmuPtr,bmu)
    #allocate space for bmu index
    bmuIndex = np.zeros(somMatrixRows).astype(np.int32)
    bmuIndexPtr = pycuda.mem_alloc(bmuIndex.nbytes)
    pycuda.memcpy_htod(bmuIndexPtr,bmuIndex)
    
    iteration=np.float32(0)
    if COLOR_MAP:    
        plt.figure(3)
        ax1 = plt.subplot(111) 
        ax1.set_title("Initial SOM")
        somplot = plt.imshow(somMatrix,interpolation="none")
        plt.show()
        plt.pause(.1)
    print "-- Starting SOM training..."
    while iteration <= maxIterations:
        sigma = sigma0 * np.exp(-iteration/tau1)
        eta = eta0 * np.exp(-iteration/tau2)
        #select a random sample
        fIndex = random.randint(0,len(featuresForSOM)-1)
            
        feats = featuresForSOM.loc[fIndex].as_matrix().astype(np.float32)
        featuresPtr = pycuda.mem_alloc(feats.nbytes)
        pycuda.memcpy_htod(featuresPtr,feats)
        #find the BMU
        computeBMU(somMatrixPtr, bmuPtr, bmuIndexPtr, featuresPtr, np.int32(featureCount), somBytesPerRow, somNumberOfRows, somNumberOfColumns, np.float32(MAX_FEAT), np.float32(INF),np.int32(metric),block=(blk,1,1),grid=(somNumberOfRows,1))
        pycuda.memcpy_dtoh(bmu,bmuPtr)
        pycuda.memcpy_dtoh(bmuIndex,bmuIndexPtr)
        minVal = np.min(bmu)
        if minVal > MAX_FEAT:
            print "*** Feature ignored in train - INVESTIGATE!!"
            continue
        block = np.argmin(bmu)
        thread = bmuIndex[block] 
        #print "In TrainSOM(). block: %i thread: %i" % (block,thread)       
        adjustWeights(somMatrixPtr, featuresPtr, np.int32(featureCount), somBytesPerRow, somNumberOfRows, somNumberOfColumns, np.float32(MAX_FEAT), np.int32(block), np.int32(thread), np.float32(sigma), np.float32(eta),  block=(blk,1,1),grid=(somNumberOfRows,1))
        pycuda.memcpy_dtoh(somMatrix,somMatrixPtr)

        #update the BMU and neighborhood nodes
        if trackTraining and not COLOR_MAP and iteration % iterPlotPoints == 0:
            avgDist = computeAvgDistancetoBMU(iteration,iterationDistance, features, nodeHitMatrix, somMatrixPtr, somMatrix, featureStatsMatrix, featuresPtr, np.int32(featureCount),  somBytesPerRow, somNumberOfRows, somNumberOfColumns)
            pycuda.memcpy_htod(somMatrixPtr,somMatrix)
            percentHitCount = 100.0*(float(len(hitCountDict.keys()))/(somNumberOfColumns*somNumberOfRows))
            vals = hitCountDict.values()
            vHits = [v[0] for v in vals]
            unique = [1 for v in vals if v[0] == 1]
            oneHits.append(len(unique))
            maxHits.append(max(vHits))
            distinctHits.append(len(hitCountDict.keys()))
            hitCountPercent.append(percentHitCount)
            fig1 = plt.figure(1,figsize=(4*3.13,6*3.13))
            fig1.subplots_adjust(wspace=1.5)
            fig1.suptitle('Training Progress', fontsize=20)
            ax1 = plt.subplot(611)
            ax1.set_ylabel('Avg Distance to BMU')
            x = [x for x in iterationCount]
            y = [y for y in iterationDistance]
            plt.plot(x,y,'r')
            ax2 = plt.subplot(612)
            ax2.set_ylabel('% of Nodes Hit')
            y1 = [y for y in hitCountPercent]
            plt.plot(x,y1,'r')
            ax3 = plt.subplot(613)
            ax3.set_ylabel('Num Single Hits')
            y2 = [y for y in oneHits]
            plt.plot(x,y2,'g')
            ax4 = plt.subplot(614)
            ax4.set_ylabel('Max Hits on Single Node')
            y3 = [y for y in maxHits]
            plt.plot(x,y3,'b')
            ax5 = plt.subplot(615)
            ax5.set_ylabel('Num of Nodes Hit')
            y4 = [y for y in distinctHits]
            plt.plot(x,y4,'y')
            fig2 = plt.figure(2,figsize=(6*3.13,4*3.13))
            fig2.suptitle('Train Node Hit Counts. Black: 0 Green: 1 Blue: <=10 Red: <=100 White >100', fontsize=20)
            ax4 = plt.subplot(111)
            somplot = plt.imshow(nodeHitMatrix,interpolation="none")
            plt.show()
            plt.pause(0.1)

        if COLOR_MAP and iteration % 1000 == 0 and iteration != maxIterations:
            pycuda.memcpy_dtoh(somMatrix,somMatrixPtr)
            plt.figure(2)
            ax1 = plt.subplot(111) 
            ax1.set_title("Iteration="+str(iteration))
            somplot = plt.imshow(somMatrix,interpolation="none")
            plt.show()
            plt.pause(0.1)  
        iteration += 1
    
    if COLOR_MAP:
        elapsed = timeit.default_timer() - start_time
        print "Elapsed time to train the SOM for %i iterations: %f secs\n" % (iteration-1,elapsed)
        pycuda.memcpy_dtoh(somMatrix,somMatrixPtr)
        plt.figure(2)
        plt.ioff()
        ax1 = plt.subplot(111)
        ax1.set_title("Final Map. Iterations="+str(iteration)+" ("+str(somMatrixRows)+","+str(somMatrixColumns)+")")
        somplot = plt.imshow(somMatrix,interpolation="none")
        plt.show()
        plt.pause(0.1)
    if not COLOR_MAP:    
        denormalize(somMatrixPtr, featureStatsMatrixPtr, np.int32(featureCount), somBytesPerRow, somNumberOfRows, somNumberOfColumns, featureStatsBytesPerRow, featureStatsNumberOfRows, np.int32(1), np.int32(0), block=(blk,1,1),grid=(somNumberOfRows,1))
        pycuda.memcpy_dtoh(somMatrix,somMatrixPtr)
        if trackTraining:
            print "SOM training complete. Average distance to BMU: %f Percent of Nodes Hit: %f " % (avgDist,percentHitCount)
            fig1.savefig('./TrainedSOMProgression_' + iterStr + '.png')
            fig2.savefig('./TrainedSOMNodeHitCountsTrainData_' + iterStr + '.png')
        elapsed = timeit.default_timer() - start_time
        print "Elapsed time to train the SOM for %i iterations: %f secs\n" % (iteration-1,elapsed)
    return somMatrix

def evaluateTrainingData(features,rv='Baseline'):
    print "Evaluating the training data..."
    start_time = timeit.default_timer()
    r = [[[0.0 for k in range(0,3)] for i in range(somMatrixColumns)] for j in range (somMatrixRows)] 
    nodeHitMatrix = np.array(r).astype(np.float32)
    nodeHitMatrix.fill(0)
    hitCountDict.clear()

    #remove non-data columns
    featuresForSOM = features.drop(omitCols,axis=1)
    featuresForSOM.fillna(2.0*MAX_FEAT,inplace=True)

    filename = './TrainedSOM_' + iterStr + '.npy'
    somMatrix = np.load(filename)
    #allocate memory for the somcuda on the device
    somMatrixPtr = pycuda.mem_alloc(somMatrix.nbytes)
    somBytesPerRow = np.int32(somMatrix.strides[0])
    somNumberOfRows = np.int32(somMatrix.shape[0])
    somNumberOfColumns = np.int32(somMatrix.shape[1])
    pycuda.memcpy_htod(somMatrixPtr,somMatrix)
    #allocate space for bmu
    bmu = np.zeros(somMatrixRows).astype(np.float32)
    bmuPtr = pycuda.mem_alloc(bmu.nbytes)
    pycuda.memcpy_htod(bmuPtr,bmu)
    #allocate space for bmu index
    bmuIndex = np.zeros(somMatrixRows).astype(np.int32)
    bmuIndexPtr = pycuda.mem_alloc(bmuIndex.nbytes)
    pycuda.memcpy_htod(bmuIndexPtr,bmuIndex)
    for i in range (0,len(featuresForSOM)):
        featuresMatrix = featuresForSOM.loc[i].as_matrix().astype(np.float32)
        featuresPtr = pycuda.mem_alloc(featuresMatrix.nbytes)
        pycuda.memcpy_htod(featuresPtr,featuresMatrix)

        #find the BMU
        computeBMUMapping(somMatrixPtr, bmuPtr, bmuIndexPtr, featuresPtr, np.int32(len(featuresForSOM.columns)),  somBytesPerRow, somNumberOfRows, somNumberOfColumns, np.float32(MAX_FEAT), np.float32(INF),np.int32(metric),block=(blk,1,1),grid=(somNumberOfRows,1))
        pycuda.memcpy_dtoh(bmu,bmuPtr)
        pycuda.memcpy_dtoh(bmuIndex,bmuIndexPtr)
        block = np.argmin(bmu)
        thread = bmuIndex[block]
        data = somMatrix[block][thread]
        if trackTraining:
            val = hitCountDict[(block,thread)]
            if val == None or len(val) == 0:
                hitCountDict[(block,thread)] = [1,i]
            else:
                hitCountDict[(block,thread)][0] += 1
            val = np.int32(hitCountDict[(block,thread)])[0]
            if val == 1:
                val = 0x0000ff00
            elif val <= 10:
                val = 0x000000ff
            elif val <= 100:
                val = 0x00ff0000
            else:
                val = 0x00ffffff
            bval = (val & 0x000000ff)
            gval = ((val & 0x0000ff00) >> 8)
            rval = ((val & 0x00ff0000) >> 16)
            nodeHitMatrix[block][thread] = [rval/255.0,gval/255.0,bval/255.0]
            if i > 0 and i % 1000 == 0:
                fig3 = plt.figure(3,figsize=(6*3.13,4*3.13))
                fig3.suptitle('Evaluation Node Hit Counts. Black: 0 Green: 1 Blue: <=10 Red: <=100 White >100', fontsize=20)
                ax = plt.subplot(111)
                somplot = plt.imshow(nodeHitMatrix,interpolation="none")
                plt.show()
                plt.pause(0.1)
    
    elapsed = timeit.default_timer() - start_time
    print "Evaluation completed in %f secs" % elapsed
    #identify and sort each dependent feature's contribution to the objective value
    if trackTraining:
        fig3.savefig('./TrainedSOMNodeHitCountsEvaluateData_' + iterStr + '.png')       
    return



#Set the action associated with the desired processing
#action = Process.EvaluateTrainingData
action = Process.ColorSOMDemo
#action = Process.TrainSOM

trackTraining = True 
normalizeMean = False
normalizeVar = False
    
#set the training parameters
metric = Metric.Euclidean
maxIterations = np.float32(10000)
somMatrixRows = 256 #maps to blocks in the kernel
somMatrixColumns = 256 #maps to threads in the kernel
iterPlotPoints = np.int32(.1*maxIterations) #not part of training parameters - just provides feedback during training if trackTraining == 1
iterStr = str(somMatrixRows)+'_'+str(somMatrixColumns)+'_'+str(np.int32(maxIterations))+"_"+str(metric)
sigma0 = np.sqrt(somMatrixRows*somMatrixRows+somMatrixColumns*somMatrixColumns) #initial neighborhood radius
tau1 = maxIterations/np.log10(sigma0) #sigma time constant
tau2 = maxIterations #eta time constant
eta0 = .1 #initial learning rate

random.seed(8756) #keep it reproducible
np.random.seed(8756)
#python hooks to CUDA kernels
computeBMU = somcuda.som_mod.get_function("computeBMUGeneric")
computeBMUMapping = somcuda.som_mod.get_function("computeBMUMapping")
adjustWeights = somcuda.som_mod.get_function("adjustWeightsGeneric")
denormalize = somcuda.som_mod.get_function("denormalize")
#ensure this network is soluable
blk = np.power(2,np.ceil(np.log2(somMatrixColumns))).astype(np.int32)  
if somMatrixColumns > MAX_THREADS_PER_BLOCK or somMatrixRows > MAX_BLOCKS_PER_GRID:
    print "******* ERROR. Program supports a maximum of %i rows and columns. rows = %i columns = %i is too large." % (MAX_THREADS_PER_BLOCK,somMatrixRows,somMatrixColumns)
    sys.exit()
option = 'Enter the desired integer associated with the runtime option below: \n'
option += str(Process.ColorSOMDemo) + ' ---Train SOM on Red, Green, Blue, Black and White RGB color data ' +  '\n'
option += str(Process.TrainSOM) +  ' ---Train SOM on csv-formated feature data ' + '(Note this file must be stored as: ./train.csv)'
print option
sys.stdout.flush()
try:
    action = int(raw_input()) 
except:
    print "\n****** Exception evaluating option. Program will terminate."
    sys.exit()   
if action == Process.EvaluateTrainingData:
    train = './train.csv'
    omitCols = ['Id']
    print "Loading the training data..."
    features = pd.read_csv(train)
    hitCountDict = defaultdict(list)
    plt.ion() # enable interactive mode   
    evaluateTrainingData(features)
    print "Program terminated normally."
    sys.exit()
elif action == Process.ColorSOMDemo: #hello world of SOMs
    train = '../data/ColorRGBBlW.csv'
    iterStr += '_ColorDemo'
    omitCols = ['Color']
    COLOR_MAP = True
    print "Loading the training data..."
    features = pd.read_csv(train)
    featuresForSOM = features.drop(omitCols,axis=1)
    featuresForSOM, featureStatsMatrix, somMatrix = initializeData()    
    iterationDistance =  []
    iterationCount = []
    hitCountDict = defaultdict(list)
    plt.ion() # enable interactive mode   
    trainSOM(features,somMatrix)
    sys.exit()
elif action == Process.TrainSOM:
    train = './train.csv'
    omitCols = ['Id']
    print "Loading the training data..."
    features = pd.read_csv(train)
    featuresForSOM = features.drop(omitCols,axis=1)
    featuresForSOM, featureStatsMatrix, somMatrix = initializeData()    
    iterationDistance =  []
    iterationCount = []
    hitCountDict = defaultdict(list)
    plt.ion() # enable interactive mode   
    somMatrix = trainSOM(features, somMatrix)
    filename = './TrainedSOM_' + iterStr +'.npy'
    np.save(filename,somMatrix)
    #evaluate objective function
    evaluateTrainingData(features)
    hitCountDict = defaultdict(list)
    print "Program terminated normally."
    sys.exit()
else:
    print "\n ******* Invalid option. Program terminating with option error." 
    sys.exit()




