#include "ComputationsBarnesHutCuda.h"
#include <thrust/sort.h>
#include <bitset>
#include <string>

const double G = 6.674 * (1e-11);
const double EPS = 0.01f;
const double theta = 0.5;
const int THREADS_PER_BLOCK = 512;
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
void calculateMortonCodes(T* positions, unsigned long long* codes, int numberOfBodies, T* mins, T* maxs) {
  int thid = blockIdx.x*blockDim.x + threadIdx.x;
  if(thid >= numberOfBodies) return;
  double t[3] = {mins[0], mins[1], mins[2]};
  double p[3] = {positions[3*thid], positions[3*thid+1], positions[3*thid+2]};
  double b[3] = {(maxs[0] - mins[0])/2, (maxs[1] - mins[1])/2, (maxs[2] - mins[2])/2};
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
    int N, int previousAllChildrenCount, int* sortedNodes, double* positions, double* weights, int level) {
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
void computeForces(OctreeNode* octree, double* velocities, double* weights, 
    double* pos, double* mins, double* maxs, int AllNodes, int N, double dt) 
{
    int thid = blockIdx.x*blockDim.x + threadIdx.x;
    if(thid >= N) return;

    double p[3] = {pos[3*thid], pos[3*thid + 1], pos[3*thid + 2]};
    int multipliers[8][3] = {{0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1}, {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1}};
    double forces[3] = {0.0, 0.0, 0.0}; 
    unsigned int stack[21];
    unsigned int child[21];
    double b[6*21];
    int top = -1;
    stack[++top] = AllNodes - 1;
    child[top] = 0;
    for(int i=0; i<3; i++) {
        b[top*6 + 2*i] = mins[i];
        b[top*6 + 2*i + 1] = maxs[i];
    }

    while(top>=0) 
    {
        int prevTop = top;
        int nextChild = child[top];
        int idx = stack[top--];
        if(idx == -1) 
            continue;
        
        if(octree[idx].position == -1)
        {            
            bool res = true;
            if(!(pos[thid*3] >= b[6*prevTop] && pos[thid*3] <= b[6*prevTop + 1]) && res) res = false;
            if(!(pos[thid*3 + 1] >= b[6*prevTop + 2] && pos[thid*3 + 1] <= b[6*prevTop + 3]) && res) res = false;
            if(!(pos[thid*3 + 2] >= b[6*prevTop + 4] && pos[thid*3 + 2] <= b[6*prevTop + 5]) && res) res = false;

            if(res)
            {
                if(nextChild==8) {
                    continue;
                }
                stack[++top] = idx;
                child[top] = nextChild + 1;
                
                stack[++top] = octree[idx].children[nextChild];
                child[top] = 0;
                for(int i=0; i<3; i++) {
                    if(!multipliers[nextChild][i]) {
                        b[top*6 + 2*i] = b[prevTop*6 + 2*i];
                        b[top*6 + 2*i + 1] = b[prevTop*6 + 2*i] + (b[prevTop*6 + 2*i + 1] - b[prevTop*6 + 2*i])/2;
                    } else {
                        b[top*6 + 2*i] = b[prevTop*6 + 2*i] + (b[prevTop*6 + 2*i + 1] - b[prevTop*6 + 2*i])/2;
                        b[top*6 + 2*i + 1] = b[prevTop*6 + 2*i + 1];
                    }
                }
                continue;
            }
            
            double TotalMass = octree[idx].totalMass;

            double distX = octree[idx].centerX/TotalMass - p[0];
            double distY = octree[idx].centerY/TotalMass - p[1];
            double distZ = octree[idx].centerZ/TotalMass - p[2];
            double dist = distX*distX + distY*distY + distZ*distZ + EPS*EPS;
            dist = dist * sqrt(dist);
            //double s = max(max(b[6*prevTop+1] - b[6*prevTop], b[6*prevTop+3] - b[6*prevTop+2]), b[6*prevTop+5] - b[6*prevTop+4]);
            //double s = ((b[6*prevTop + 1] - b[6*prevTop]) + (b[6*prevTop + 3] - b[6*prevTop + 2]) + (b[6*prevTop + 5] - b[6*prevTop + 4]))/3;
            double s = b[6*prevTop+1] - b[6*prevTop];
            bool isFarAway = (s/dist < theta);
            
            if(isFarAway)
            {
                double F = G * (octree[idx].totalMass * weights[thid]);
                forces[0] += F * distX / dist;
                forces[1] += F * distY / dist;
                forces[2] += F * distZ / dist;
            }
            else
            { 
                if(nextChild==8) {
                    continue;
                }
                stack[++top] = idx;
                child[top] = nextChild + 1;

                stack[++top] = octree[idx].children[nextChild];
                child[top] = 0;
                for(int i=0; i<3; i++) {
                    if(!multipliers[nextChild][i]) {
                        b[top*6 + 2*i] = b[prevTop*6 + 2*i];
                        b[top*6 + 2*i + 1] = b[prevTop*6 + 2*i] + (b[prevTop*6 + 2*i + 1] - b[prevTop*6 + 2*i])/2;
                    } else {
                        b[top*6 + 2*i] = b[prevTop*6 + 2*i] + (b[prevTop*6 + 2*i + 1] - b[prevTop*6 + 2*i])/2;
                        b[top*6 + 2*i + 1] = b[prevTop*6 + 2*i + 1];
                    }
                }
                continue;                
            }
        }
        else 
        {
            int p = octree[idx].position;
            
            if(thid == p) 
                continue;
            
            // sortedNodes[3*idx] zamiast 3*idx?
            double distX = pos[3*p] - pos[3*thid];
            double distY = pos[3*p + 1] - pos[3*thid + 1];
            double distZ = pos[3*p + 2] - pos[3*thid + 2];
            double dist = (distX * distX + distY * distY + distZ * distZ) + EPS * EPS;
            dist = dist * sqrt(dist);
            double F = G * (weights[p] * weights[thid]);
            forces[0] += F * distX / dist; 
            forces[1] += F * distY / dist;
            forces[2] += F * distZ / dist;
        }
    }
    //if(thid >= 8 && thid <=10)
    //printf("%d : %f %f %f\n", thid, forces[0], forces[1], forces[2]);
    
    // sortedNodes[thid] zamiast thid?
    for (int j = 0; j < 3; j++) {
        double acceleration = forces[j] / weights[thid];
        pos[thid * 3 + j] +=
            velocities[thid * 3 + j] * dt + acceleration * dt * dt / 2;
        velocities[thid * 3 + j] += acceleration * dt;
    } 
}

void ComputationsBarnesHut::createTree(int numberOfBodies, double dt) {
    int blocks = (numberOfBodies+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK;
    // 0. liczymy boundaries
    // ew. zaokraglic do najblizszej potegi 2 (najwiekszej dla kazdego z wymiarow)
    thrust::device_vector<double> px(numberOfBodies), py(numberOfBodies), pz(numberOfBodies);
    double *d_px = thrust::raw_pointer_cast(px.data());
    double *d_py = thrust::raw_pointer_cast(py.data());
    double *d_pz = thrust::raw_pointer_cast(pz.data());
    
    getDimensions<<<blocks, THREADS_PER_BLOCK>>>(d_positions, d_px, d_py, d_pz, numberOfBodies);
    
    auto itx = thrust::minmax_element(px.begin(), px.end());
    auto ity = thrust::minmax_element(py.begin(), py.end());
    auto itz = thrust::minmax_element(pz.begin(), pz.end());
    double mins[3] = {*itx.first, *ity.first, *itz.first};
    double maxs[3] = {*itx.second, *ity.second, *itz.second};
    
    thrust::device_vector<double> minsDeviceVector(mins, mins+3);
    thrust::device_vector<double> maxsDeviceVector(maxs, maxs+3);
    double* d_mins = thrust::raw_pointer_cast(minsDeviceVector.data());
    double* d_maxs = thrust::raw_pointer_cast(maxsDeviceVector.data());
    
    // 1. liczymy kody mortona
    thrust::device_vector<unsigned long long> mortonCodes(numberOfBodies);
    unsigned long long* d_codes = thrust::raw_pointer_cast(mortonCodes.data());
    calculateMortonCodes<<<blocks, THREADS_PER_BLOCK>>>(d_positions, d_codes, numberOfBodies, d_mins, d_maxs); 
    
    thrust::device_vector<int> sortedNodes(numberOfBodies);
    int* d_sortedNodes = thrust::raw_pointer_cast(sortedNodes.data());
    fillNodes<<<blocks, THREADS_PER_BLOCK>>>(d_sortedNodes, numberOfBodies);

    // 2. sortujemy to
    thrust::sort_by_key(mortonCodes.begin(), mortonCodes.end(), sortedNodes.begin());  

    // 3. usuwamy duplikaty
    //auto iterators = thrust::unique_by_key(mortonCodes.begin(), mortonCodes.end(), sortedNodes.begin());
    //mortonCodes.erase(iterators.first, mortonCodes.end());
    //sortedNodes.erase(iterators.second, sortedNodes.end());

    // 5. liczymy ilość node-ow z octree do zaalokowania
    // uwaga: być może nie musimy tego tak naprawde liczyć tylko potem przy tworzeniu drzewa to się samo liczy o.o
    int uniquePointsCount = mortonCodes.size(); //thrust::distance(mortonCodes.begin(), iterators.first);
    // printf("UNIQUE POINTS COUNT %d\n", uniquePointsCount);
    blocks = (uniquePointsCount+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK;

    thrust::device_vector<OctreeNode> octree(uniquePointsCount);
    OctreeNode* d_octree = thrust::raw_pointer_cast(octree.data());
	
    // 6. laczymy octree nodes
    thrust::device_vector<int> parentsNumbers(uniquePointsCount);
    int* d_parentsNumbers = thrust::raw_pointer_cast(parentsNumbers.data());
    int childrenCount = uniquePointsCount;
    int allChildrenCount = uniquePointsCount;
    int previousAllChildrenCount = 0;
    
    for(int i = 0; i < K; ++i) {
        //printf("Aha... %d %d\n", i, allChildrenCount);
        blocks = (childrenCount+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK;
        thrust::fill(parentsNumbers.begin(), parentsNumbers.end(), 0);
        calculateDuplicates<<<blocks, THREADS_PER_BLOCK>>>(
            d_codes,
            d_parentsNumbers,
            childrenCount
        );
        
        thrust::inclusive_scan(parentsNumbers.begin(), parentsNumbers.end(), parentsNumbers.begin());
        octree.insert(octree.end(), parentsNumbers[childrenCount-1]+1, OctreeNode()); // co tu sie odbywa?
        d_octree = thrust::raw_pointer_cast(octree.data()); // dlaczego znowu raw_cast?
        
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
    blocks = (uniquePointsCount+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK;

    double *d_velocities = thrust::raw_pointer_cast(veloD.data());
    double *d_weights = thrust::raw_pointer_cast(weightsD.data());
    
    computeForces<<<blocks, THREADS_PER_BLOCK>>>(d_octree,
        d_velocities, 
        d_weights, 
        d_positions, 
        d_mins, d_maxs,
        allChildrenCount, 
        numberOfBodies, dt); 
}

bool ComputationsBarnesHut::testingMomemntum(int numberOfBodies) {
    double momentum[3] = {0.0f, 0.0f, 0.0f};
    for (unsigned i = 0; i < numberOfBodies; i++) {
        for(int k = 0; k < 3; k++) {
            momentum[k] += (weightsD[i] * veloD[i*3 + k]);
        }
    }
    std::cout << momentum[0] << " " << momentum[1] << " " << momentum[2] << std::endl;
    return true;
  }

void ComputationsBarnesHut::BarnesHutBridge(type &pos, int numberOfBodies, double dt) {
    thrust::device_vector<double> posD = pos;
    d_positions = thrust::raw_pointer_cast(posD.data());
    createTree(numberOfBodies, dt);
    //testingMomemntum(numberOfBodies);
    pos = posD;
}