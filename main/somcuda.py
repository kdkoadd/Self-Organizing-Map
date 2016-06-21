'''
Created on Jan 10, 2016

@author: Mick Hart
'''
import pycuda.autoinit
import pycuda.driver as pycuda
from pycuda.compiler import SourceModule

# CUDA code
som_mod = SourceModule("""
#include <stdio.h>
#define MAX_THREADS_PER_BLOCK 512
#define MAX_FEATURES 256
#define EUCLIDEAN 0
#define ABSOLUTE_VALUE 1
////////////////////////////////////////////////////////////////////////////////////////////////////////
//Code herein assumes compute capability 5.2: maximum number of threads per block: 1024, maximum number of x-dimension blocks: 1024, etc.
//THIS CODE DOES NOT INTERROGATE DEVICE CAPABILITIES AND ADJUST ACCORDINGLY
///////////////////////////////////////////////////////////////////////////////////////////////////////

//denormalizes the feature data after training.
__global__ void denormalize(float *som, float *featureStats, int featureCount, int somBytesPerRow, int somNumberOfRows, int somNumberOfCols, int featureStatsBytesPerRow,  int featureStatsNumberOfRows, int sigmaOffset, int muOffset)
{
    __shared__ float mu[MAX_FEATURES];
    __shared__ float sigma[MAX_FEATURES];
    
    if(threadIdx.x >= somNumberOfCols || blockIdx.x >= somNumberOfRows)
        return;

    if(threadIdx.x == 0)
    {
        for(int i=0;i<featureCount;i++)
        {
            mu[i] = *((float*)((char *)featureStats + muOffset*featureStatsBytesPerRow) + i*featureStatsNumberOfRows);
            sigma[i] = *((float*)((char *)featureStats + sigmaOffset*featureStatsBytesPerRow) + i*featureStatsNumberOfRows);
        }
    }
    __syncthreads();
    
    float *somFeatureBase = (float *)((char *)som + blockIdx.x*somBytesPerRow) + threadIdx.x*featureCount;
    for(int i=0;i<featureCount;i++)
    {                                
        float *somFeature = somFeatureBase + i;
        *somFeature = *somFeature*sigma[i] + mu[i];
    }    
}
//maps feature data onto a trained SOM
__global__ void computeBMUMapping(float *som, float *bmu, int *bmu_index, float *features, int featureCount, int somBytesPerRow, int somNumberOfRows, int somNumberOfCols, float maxFeat, float inf, int metric)
{
    __shared__ float feature[MAX_FEATURES];
    __shared__ float nodeMetric[MAX_THREADS_PER_BLOCK];
    __shared__ float nodeIndex[MAX_THREADS_PER_BLOCK];

    if(blockIdx.x >= somNumberOfRows || threadIdx.x >= somNumberOfCols)
    {
        nodeMetric[threadIdx.x] = inf;
        return;
    }
    
    nodeMetric[threadIdx.x] = 0.0;
    nodeIndex[threadIdx.x] = threadIdx.x;

    if(threadIdx.x == 0)
    {
        for(int i=0;i<featureCount;i++)
            feature[i] = features[i];
    }
    __syncthreads();  
    
    //compute metric -- Euclidean distance
    float *somFeatureBase = (float *)((char *)som + blockIdx.x*somBytesPerRow) + threadIdx.x*(featureCount);
    for(int i=0;i<featureCount;i++) //omit dependent vars from bmu calculation
    {
        if(feature[i] > maxFeat) //omit missing data - assumption is missing data values will be set outside this range by host
        {
            continue;
        }
        float somFeature = *(somFeatureBase + i);
        if(metric == EUCLIDEAN)
            nodeMetric[threadIdx.x] += (somFeature - feature[i])*(somFeature - feature[i]);
        else if(metric == ABSOLUTE_VALUE)
            nodeMetric[threadIdx.x] += abs(somFeature - feature[i]);
    }
    if(metric == EUCLIDEAN)
        nodeMetric[threadIdx.x] = sqrt(nodeMetric[threadIdx.x]);   
    __syncthreads();
    
    //find minimum metric amongst all in this block 
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) 
    {
        if (threadIdx.x < s)
        {
            nodeMetric[threadIdx.x] = min(nodeMetric[threadIdx.x], nodeMetric[threadIdx.x+s]);
            if(nodeMetric[threadIdx.x] == nodeMetric[threadIdx.x+s])
            {
                nodeIndex[threadIdx.x] = nodeIndex[threadIdx.x + s];
            }    
        }
        __syncthreads();
    }
    __syncthreads();
    
    if(threadIdx.x == 0)
    {
        bmu[blockIdx.x] = nodeMetric[threadIdx.x]; //save the minimum BMU in this block to global memory
        bmu_index[blockIdx.x] = nodeIndex[threadIdx.x]; //save the index into the som blockIdx.x row corresponding to the BMU value
    }    
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//computeBMUGeneric() identifies the best matching unit (BMU) in the SOM network for the input feature vector.
//som is a GPU global memory pointer to mxn-dimensional matrix of the SOM network.
//This kernel should be launched with n x-dimension threads and m x-dimension blocks; so,
//each block of threads corresponds to a "row" of nodes in the SOM network
//all threads train on the same feature vector concurrently with each thread computing its distance from same; consequently,
//this kernel relies on CPU synchronization of blocks. Each block will compute its BMU by minimizing the distance metric computed by 
//its associated threads. This value is stored to global memory indexed by the corresponding blockID. 
//The host will then implicitly block on a read of this global memory space (again representing tthe BMU within each block), and identify
//the minimum BMU amongst the block-computed BMUs to establish the global BMU.
//som: pointer to the mxn-dimensional SOM matrix
//bmu: BMU value for the given thread block
//bum_index: index/offset of the node within the block corresponding to the minimum BMU value 
//features: pointer to the current feature set over which the SOM is being trained
//featureCount: number of features in the training data
//somBytesPerRow: total number of bytes in a row - provides blockIdx.x-based offset into som matrix
//somNumberOfRows: number of rows in the som matrix
//somNumberOfCols: number of columns in the som Matrix
//inf: very large number. probably not necessary but keeps Python and CUDA consistent with "inf" values
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void computeBMUGeneric(float *som, float *bmu, int *bmu_index, float *features, int featureCount, int somBytesPerRow, int somNumberOfRows, int somNumberOfCols, float maxFeat, float inf, int metric)
{
    __shared__ float feature[MAX_FEATURES];
    __shared__ float nodeMetric[MAX_THREADS_PER_BLOCK];
    __shared__ float nodeIndex[MAX_THREADS_PER_BLOCK];

    if(blockIdx.x >= somNumberOfRows || threadIdx.x >= somNumberOfCols)
    {
        nodeMetric[threadIdx.x] = inf;
        return;
    }
    
    nodeMetric[threadIdx.x] = 0.0;
    nodeIndex[threadIdx.x] = threadIdx.x;

    if(threadIdx.x == 0)
    {
        for(int i=0;i<featureCount;i++)
            feature[i] = features[i];
    }
    __syncthreads();  
    
    //compute metric
    float *somFeatureBase = (float *)((char *)som + blockIdx.x*somBytesPerRow) + threadIdx.x*featureCount;
    for(int i=0;i<featureCount;i++) 
    {
        if(feature[i] > maxFeat) //omit missing data - assumption is missing data values will be set outside this range by host
        {
            continue;
        }
        float somFeature = *(somFeatureBase + i);
        if(metric == EUCLIDEAN)
            nodeMetric[threadIdx.x] += (somFeature - feature[i])*(somFeature - feature[i]);
        else if(metric == ABSOLUTE_VALUE)
            nodeMetric[threadIdx.x] += abs(somFeature - feature[i]);
    }
    if(metric == EUCLIDEAN)
        nodeMetric[threadIdx.x] = sqrt(nodeMetric[threadIdx.x]);   
    __syncthreads();
    
    //find minimum metric amongst all in this block 
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) 
    {
        if (threadIdx.x < s)
        {
            nodeMetric[threadIdx.x] = min(nodeMetric[threadIdx.x], nodeMetric[threadIdx.x+s]);
            if(nodeMetric[threadIdx.x] == nodeMetric[threadIdx.x+s])
            {
                nodeIndex[threadIdx.x] = nodeIndex[threadIdx.x + s];
            }    
        }
        __syncthreads();
    }
    __syncthreads();
    
    if(threadIdx.x == 0)
    {
        bmu[blockIdx.x] = nodeMetric[threadIdx.x]; //save the minimum BMU in this block to global memory
        bmu_index[blockIdx.x] = nodeIndex[threadIdx.x]; //save the index into the som blockIdx.x row corresponding to the BMU value
    }    
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//adjustWeightsGeneric() updates the weights for all nodes utilizing the identified best matching unit (BMU), and the current learning rates (eta) and neighborhood function (hij)
//som: pointer to the mxn-dimensional SOM matrix
//features: pointer to the current feature set over which the SOM is being trained
//featureCount: number of features in the training data
//somBytesPerRow: total number of bytes in a row - provides blockIdx.x-based offset into som matrix
//somNumberOfRows: number of rows in the SOM matrix
//somNumberOfCols: number of columns in the SOM Matrix
//(bmuBlock,bmuThread) is the SOM matrix index of the identified BMU
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void adjustWeightsGeneric(float *som, float *features, int featureCount, int somBytesPerRow, int somNumberOfRows, int somNumberOfCols, float maxFeat, int bmuBlock, int bmuThread, float sigma, float eta)
{
    __shared__ float feature[MAX_FEATURES];
    __shared__ float twoSigmaSigma;
    float distanceToBMU = 0.0;
    float hij = 0.0;
    
    if(threadIdx.x >= somNumberOfCols || blockIdx.x >= somNumberOfRows)
        return;
    
    if(threadIdx.x == 0)
        twoSigmaSigma = 2.0*sigma*sigma;
        
    if(threadIdx.x == 0)
    {
        for(int i=0;i<featureCount;i++)
            feature[i] = features[i];
    }
    __syncthreads();
    
    //compute distance between geometric locations of BMU and this node - sqrt omitted since squared when used for hij calculation
    //distanceToBMU = (bmuBlock-blockIdx.x)*(bmuBlock-blockIdx.x) + (bmuThread-threadIdx.x)*(bmuThread-threadIdx.x);
    int x1 = abs((int)(bmuBlock-blockIdx.x));
    int x2 = somNumberOfCols - abs((int)(bmuBlock-blockIdx.x));
    int x = min(x1,x2);
    int y1 = abs((int)(bmuThread-threadIdx.x));
    int y2 = somNumberOfRows - abs((int)(bmuThread-threadIdx.x));
    int y = min(y1,y2);
    distanceToBMU = x*x + y*y;

    //compute the neighborhood function
    hij = exp(-distanceToBMU/(twoSigmaSigma));
    float *somFeatureBase = (float *)((char *)som + blockIdx.x*somBytesPerRow) + threadIdx.x*featureCount;
    float somFeature = 0.0;
    
    for(int i=0;i<featureCount;i++)
    {
        if(feature[i] > maxFeat) //omit missing data - assumption is missing data values will be set outside this range by host
            continue;
        float *somFeatureLoc = somFeatureBase + i;
        somFeature = *somFeatureLoc;
        *somFeatureLoc = somFeature + eta*hij*(feature[i]-somFeature);
    }
}
""")
