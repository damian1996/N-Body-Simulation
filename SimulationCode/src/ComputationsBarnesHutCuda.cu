#include "ComputationsBarnesHutCuda.h"
#include <thrust/sort.h>
#include <bitset>
#include <string>

const float G = 6.674 * (1e-11);
const float EPS = 0.01f;
const float theta = 2;
const int THREADS_PER_BLOCK = 1024;
const int K = 15;

struct OctreeNode {
  int children[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
  int position = -1;
  float totalMass = 0;
  float centerX = 0.0;
  float centerY = 0.0;
  float centerZ = 0.0;
};

template <typename T>
__global__
void getDimensions(T* positions, T* px, T* py, T* pz, int numberOfBodies) {
  int thid = blockIdx.x*blockDim.x + threadIdx.x;
  if(thid >= numberOfBodies) return;
  px[thid] = positions[3*thid];
  py[thid] = positions[3*thid+1];
  pz[thid] = positions[3*thid+2];
}

template <typename T>
__global__
void calculateMortonCodes(T* positions, unsigned long long* codes, int numberOfBodies, float* mins, float* maxs) {
  int thid = blockIdx.x*blockDim.x + threadIdx.x;
  if(thid >= numberOfBodies) return;
  float t[3] = {mins[0], mins[1], mins[2]};
  float p[3] = {positions[3*thid], positions[3*thid+1], positions[3*thid+2]};
  float b[3] = {(maxs[0] - mins[0])/2, (maxs[1] - mins[1])/2, (maxs[2] - mins[2])/2};
  unsigned long long code = 0;
  for(int i = 0; i < K; ++i) {
    for(int j = 0; j < 3; ++j) {
      code <<= 1;
      if(t[j]+b[j] < p[j]) {
        code |= 0x1;
        t[j] += b[j];
      }
      b[j] /= 2;
    }
  }
  codes[thid] = code;
}

template <typename T>
__global__
void fillNodes(T* sortedNodes, int numberOfBodies) {
  int thid = blockIdx.x*blockDim.x + threadIdx.x;
  if(thid >= numberOfBodies) return;
  sortedNodes[thid] = thid;
}

__global__
void calculateDuplicates(unsigned long long int* mortonCodes, int* result, int N) {
  int thid = blockIdx.x*blockDim.x + threadIdx.x;
  if(thid >= N || thid == 0) return;
  unsigned long long int code = mortonCodes[thid];
  unsigned long long int previous_code = mortonCodes[thid-1];
  code >>= 3;
  previous_code >>= 3;
  result[thid] = (code != previous_code);
}

__global__
void connectChildren(unsigned long long int* mortonCodes, int* parentsNumbers, OctreeNode* octree, 
    int N, int previousAllChildrenCount, int* sortedNodes, float* positions, float* weights, int level) {
    int thid = blockIdx.x*blockDim.x + threadIdx.x;
    if(thid >= N) return;
    unsigned long long int childNumber = mortonCodes[thid] & 0x7; // 7 = 111 binarnie
    octree[parentsNumbers[thid]].children[childNumber] = thid+previousAllChildrenCount;
    octree[parentsNumbers[thid]].position = -1;
    octree[thid+previousAllChildrenCount].position = ((level == 0) ? sortedNodes[thid] : -1);
    int childIndex = sortedNodes[thid];
    if(level == 0) {
        octree[thid].totalMass = weights[childIndex];
        octree[thid].centerX = weights[childIndex] * positions[3*childIndex];
        octree[thid].centerY = weights[childIndex] * positions[3*childIndex+1];
        octree[thid].centerZ = weights[childIndex] * positions[3*childIndex+2];
    }
    int pthid = parentsNumbers[thid];
    atomicAdd(&octree[pthid].totalMass, octree[thid+previousAllChildrenCount].totalMass);
    atomicAdd(&octree[pthid].centerX, octree[thid+previousAllChildrenCount].centerX);
    atomicAdd(&octree[pthid].centerY, octree[thid+previousAllChildrenCount].centerY);
    atomicAdd(&octree[pthid].centerZ, octree[thid+previousAllChildrenCount].centerZ);
}

__global__
void computeForces(OctreeNode* octree, float* velocities, float* weights, 
    float* pos, float* mins, float* maxs, int AllNodes, int N, float dt) 
{
    int thid = blockIdx.x*blockDim.x + threadIdx.x;
    if(thid >= N) return;

    float p[3] = {pos[3*thid], pos[3*thid + 1], pos[3*thid + 2]};
    float forces[3] = {0.0, 0.0, 0.0};
    const int C = 0;//16; 
    int stack[16];
    char child[16];
    float bound = maxs[0]-mins[0];
    int top = 0;
    stack[threadIdx.x*C+top] = AllNodes - 1;
    child[threadIdx.x*C+top] = 0;

    while(top>=0) 
    {
        int prevTop = top;
        int nextChild = child[threadIdx.x*C+top];
        int idx = stack[threadIdx.x*C+top];
        top--;
        if(idx == -1) 
            continue;
        
        if(octree[idx].position == -1)
        {
            float distX = octree[idx].centerX - p[0];
            float distY = octree[idx].centerY - p[1];
            float distZ = octree[idx].centerZ - p[2];
            float dist = distX*distX + distY*distY + distZ*distZ + EPS*EPS;
            dist = dist * sqrt(dist);
            float s = bound/(1<<prevTop);
            bool isFarAway = (s/dist < theta);
            
            if(isFarAway)
            {
                float F = G * (octree[idx].totalMass * weights[thid]);
                forces[0] += F * distX / dist;
                forces[1] += F * distY / dist;
                forces[2] += F * distZ / dist;
            }
            else
            { 
                if(nextChild==8) {
                    continue;
                }
                ++top;
                stack[threadIdx.x*C+top] = idx;
                child[threadIdx.x*C+top] = nextChild + 1;

                ++top;
                stack[threadIdx.x*C+top] = octree[idx].children[nextChild];
                child[threadIdx.x*C+top] = 0;
                continue;                
            }
        }
        else 
        {
            int p = octree[idx].position;
            
            if(thid == p) 
                continue;
            
            float distX = pos[3*p] - pos[3*thid];
            float distY = pos[3*p + 1] - pos[3*thid + 1];
            float distZ = pos[3*p + 2] - pos[3*thid + 2];
            float dist = (distX * distX + distY * distY + distZ * distZ) + EPS * EPS;
            dist = dist * sqrt(dist);
            float F = G * (weights[p] * weights[thid]);
            forces[0] += F * distX / dist; 
            forces[1] += F * distY / dist;
            forces[2] += F * distZ / dist;
        }
    }
    
    for (int j = 0; j < 3; j++) {
        float acceleration = forces[j] / weights[thid];
        pos[thid * 3 + j] +=
            velocities[thid * 3 + j] * dt + acceleration * dt * dt / 2;
        velocities[thid * 3 + j] += acceleration * dt;
    } 
}

__global__
void computeCenterOfMasses(OctreeNode* octree, int N) {
    
    int thid = blockIdx.x*blockDim.x + threadIdx.x;
    if(thid >= N) return;

    int totalMass = octree[thid].totalMass;
    
    octree[thid].centerX /= totalMass;
    octree[thid].centerY /= totalMass;
    octree[thid].centerZ /= totalMass;
}

void ComputationsBarnesHut::createTree(int numberOfBodies, float dt) {
    int blocks = (numberOfBodies+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK;
    thrust::device_vector<float> px(numberOfBodies), py(numberOfBodies), pz(numberOfBodies);
    float *d_px = thrust::raw_pointer_cast(px.data());
    float *d_py = thrust::raw_pointer_cast(py.data());
    float *d_pz = thrust::raw_pointer_cast(pz.data());
    
    getDimensions<<<blocks, THREADS_PER_BLOCK>>>(d_positions, d_px, d_py, d_pz, numberOfBodies);
    
    auto itx = thrust::minmax_element(px.begin(), px.end());
    auto ity = thrust::minmax_element(py.begin(), py.end());
    auto itz = thrust::minmax_element(pz.begin(), pz.end());
    float mins[3] = {*itx.first, *ity.first, *itz.first};
    float maxs[3] = {*itx.second, *ity.second, *itz.second};
    
    thrust::device_vector<float> minsDeviceVector(mins, mins+3);
    thrust::device_vector<float> maxsDeviceVector(maxs, maxs+3);
    float* d_mins = thrust::raw_pointer_cast(minsDeviceVector.data());
    float* d_maxs = thrust::raw_pointer_cast(maxsDeviceVector.data());
    
    thrust::device_vector<unsigned long long> mortonCodes(numberOfBodies);
    unsigned long long* d_codes = thrust::raw_pointer_cast(mortonCodes.data());
    calculateMortonCodes<<<blocks, THREADS_PER_BLOCK>>>(d_positions, d_codes, numberOfBodies, d_mins, d_maxs); 
    
    thrust::device_vector<int> sortedNodes(numberOfBodies);
    int* d_sortedNodes = thrust::raw_pointer_cast(sortedNodes.data());
    fillNodes<<<blocks, THREADS_PER_BLOCK>>>(d_sortedNodes, numberOfBodies);

    thrust::sort_by_key(mortonCodes.begin(), mortonCodes.end(), sortedNodes.begin());  

    int uniquePointsCount = mortonCodes.size(); 
    blocks = (uniquePointsCount+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK;

    thrust::device_vector<OctreeNode> octree(uniquePointsCount);
    OctreeNode* d_octree = thrust::raw_pointer_cast(octree.data());

    thrust::device_vector<int> parentsNumbers(uniquePointsCount);
    int* d_parentsNumbers = thrust::raw_pointer_cast(parentsNumbers.data());
    int childrenCount = uniquePointsCount;
    int allChildrenCount = uniquePointsCount;
    int previousAllChildrenCount = 0;
    
    for(int i = 0; i < K; ++i) {
        blocks = (childrenCount+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK;
        thrust::fill(parentsNumbers.begin(), parentsNumbers.end(), 0);
        calculateDuplicates<<<blocks, THREADS_PER_BLOCK>>>(
            d_codes,
            d_parentsNumbers,
            childrenCount
        );
        
        thrust::inclusive_scan(parentsNumbers.begin(), parentsNumbers.end(), parentsNumbers.begin());
        octree.insert(octree.end(), parentsNumbers[childrenCount-1]+1, OctreeNode());
        d_octree = thrust::raw_pointer_cast(octree.data());
        
        thrust::for_each(parentsNumbers.begin(), parentsNumbers.end(), thrust::placeholders::_1 += allChildrenCount);
        
        connectChildren<<<blocks, THREADS_PER_BLOCK>>>(
            d_codes,
            d_parentsNumbers,
            d_octree,
            childrenCount,
            previousAllChildrenCount,
            d_sortedNodes,
            d_positions,
            d_weights,
            i
        );

        thrust::for_each(mortonCodes.begin(), mortonCodes.end(), thrust::placeholders::_1 >>= 3);        
        auto it = thrust::unique(mortonCodes.begin(), mortonCodes.end());
        mortonCodes.erase(it, mortonCodes.end());
        d_codes = thrust::raw_pointer_cast(mortonCodes.data()); // dlaczego znowu raw_cast?
        childrenCount = mortonCodes.size();
        previousAllChildrenCount = allChildrenCount;
        allChildrenCount += childrenCount;
    }
    blocks = (octree.size()+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK;
    computeCenterOfMasses<<<blocks, THREADS_PER_BLOCK>>>(
        d_octree,
        octree.size()
    );
    

    float *d_velocities = thrust::raw_pointer_cast(veloD.data());
    float *d_weights = thrust::raw_pointer_cast(weightsD.data());

    blocks = (numberOfBodies+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK;
    
    computeForces<<<blocks, THREADS_PER_BLOCK>>>(d_octree,
        d_velocities, 
        d_weights, 
        d_positions, 
        d_mins, d_maxs,
        allChildrenCount, 
        numberOfBodies, dt); 
}

bool ComputationsBarnesHut::testingMomemntum(int numberOfBodies) {
    float momentum[3] = {0.0f, 0.0f, 0.0f};
    for (unsigned i = 0; i < numberOfBodies; i++) {
        for(int k = 0; k < 3; k++) {
            momentum[k] += (weightsD[i] * veloD[i*3 + k]);
        }
    }
    std::cout << momentum[0] << " " << momentum[1] << " " << momentum[2] << std::endl;
    return true;
  }

void ComputationsBarnesHut::BarnesHutBridge(type &pos, int numberOfBodies, float dt) {
    thrust::device_vector<float> posD = pos;
    d_positions = thrust::raw_pointer_cast(posD.data());
    createTree(numberOfBodies, dt);
    //testingMomemntum(numberOfBodies);
    pos = posD;
}