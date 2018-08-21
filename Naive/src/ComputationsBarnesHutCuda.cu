#include "ComputationsBarnesHutCuda.h"
#include <thrust/sort.h>
#include <bitset>
#include <string>

const float G = 6.674 * (1e-11);
const float EPS = 0.01f;
const int numberOfChilds = 8;
const float theta = 0.5;
const int THREADS_PER_BLOCK = 1024;
const int K = 20;

struct OctreeNode {
  // jak sprawdzac czy node ma dzieci czy nie?
  int children[8];
  int position;
  float totalMass = 0;
  float centerX = 0;
  float centerY = 0;
  float centerZ = 0;
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
    int N, int previousChildrenCount, int* sortedNodes, int level) {
  int thid = blockIdx.x*blockDim.x + threadIdx.x;
  if(thid >= N) return;
  // dziecko łączy się z parentsNumbers[thid] pod wskaźnikiem zależnym od bitów
  unsigned long long int childNumber = mortonCodes[thid]&0x7; // 7 = 111 binarnie
  octree[parentsNumbers[thid]].children[childNumber] = thid+previousChildrenCount;
  octree[thid+previousChildrenCount].position = level == 0 ? sortedNodes[thid] : -1;
  octree[parentsNumbers[thid]].totalMass += octree[childNumber].totalMass;

  octree[parentsNumbers[thid]].centerX += cos*cos;
  octree[parentsNumbers[thid]].centerY += cos*cos;
  octree[parentsNumbers[thid]].centerZ += cos*cos;
}

__device__ 
void 

__global__
void computeForces(OctreeNode* octree, float* forces, float* velocities, float* weights, 
    float* pos, int AllNodes, int N, float dt) 
{
    // informacje czy jest punktem/czy ma dzieci (jeden z trzech stanow)
    // pozycje, centerOfMass, mass, totalMass, 
    int thid = blockIdx.x*blockDim.x + threadIdx.x;
    if(thid >= N) return;

    forces[thid] = 0.0;
    unsigned long long int stack[64];
    int top = -1;
    //stack[0] = root;
    stack[++top] = AllNodes-1;
    while(top>=0) 
    {
        int idx = stack[top--];
        if(octree[idx].position == -1)
        {
            // jakos wywalic i!=j

            double d = sqrt(x - pos[3*thid] + y - pos[3*thid+1] + z - pos[3*thid+2] + EPS*EPS); 
            double s = boundaries[1] - boundaries[0];
            bool isFarAway = (s/d < theta);
            if(isFarAway)
            {
                double distX = octree[idx].centerX - pos[0];
                double distY = octree[idx].centerY - pos[1];
                double distZ = octree[idx].centerZ - pos[2];
                double dist = (distX * distX + distY * distY + distZ * distZ) + EPS * EPS;
                dist = dist * sqrt(dist);
                float F = G * (r->getTotalMass() * weights[i]);
                forces[i * 3] += F * distX / dist;
                forces[i * 3 + 1] += F * distY / dist;
                forces[i * 3 + 2] += F * distZ / dist;
            }
            else
            {
                for(int i=0; i<numberOfChilds; i++) 
                {
                    stack[++top] = octree.children[i];
                }
            }
        }
        else 
        {
            if(thid == octree[idx].position) 
                continue;
            
            // Jesli node jest zewnetrzny, to policz sile ktora wywiera ten node na obecnie rozwazane cialo
            float distX = octree[idx].centerX - pos[3*thid];
            float distY = octree[idx].centerY - pos[3*thid + 1];
            float distZ = octree[idx].centerZ - pos[3*thid + 2];
            float dist = (distX * distX + distY * distY + distZ * distZ) + EPS * EPS;
            dist = dist * sqrt(dist);
            // jak sprawdzic czy to te same bodies? Musze zachowac i != j
            float F = G * (weights[octree[idx].position] * weights[thid]);
            forces[thid * 3] += F * distX / dist; // force = G(m1*m2)/r^2
            forces[thid * 3 + 1] += F * distY / dist;
            forces[thid * 3 + 2] += F * distZ / dist;
        }
    }


    for (int j = 0; j < 3; j++) {
        float acceleration = forces[thid * 3 + j] / weights[thid];
        positions[thid * 3 + j] +=
            velocities[thid * 3 + j] * dt + acceleration * dt * dt / 2;
        velocities[thid * 3 + j] += acceleration * dt;
    }
}

void ComputationsBarnesHut::createTree(int numberOfBodies) {
    int blocks = (numberOfBodies+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK;
    // 0. liczymy boundaries
    // ew. zaokraglic do najblizszej potegi 2 (najwiekszej dla kazdego z wymiarow)
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
    auto iterators = thrust::unique_by_key(mortonCodes.begin(), mortonCodes.end(), sortedNodes.begin());

    // 5. liczymy ilość node-ow z octree do zaalokowania
    // uwaga: być może nie musimy tego tak naprawde liczyć tylko potem przy tworzeniu drzewa to się samo liczy o.o
    int uniquePointsCount = thrust::distance(mortonCodes.begin(), iterators.first);
    blocks = (uniquePointsCount+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK;

    thrust::device_vector<OctreeNode> octree(uniquePointsCount);
    OctreeNode* d_octree = thrust::raw_pointer_cast(octree.data());
	
    // 6. laczymy octree nodes
    thrust::device_vector<int> parentsNumbers(uniquePointsCount);
    int* d_parentsNumbers = thrust::raw_pointer_cast(parentsNumbers.data());
    int childrenCount = uniquePointsCount;
    int allChildrenCount = uniquePointsCount;
    int previousChildrenCount = 0;
    
    // dzieci laczymy bottom-up czy up-down? #damian

    for(int i = 0; i < K; ++i) {
        blocks = (childrenCount+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK;
        thrust::fill(parentsNumbers.begin(), parentsNumbers.end(), 0);
        // policz tablice takich samych
        calculateDuplicates<<<blocks, THREADS_PER_BLOCK>>>(
            d_codes,
            d_parentsNumbers,
            childrenCount
        );
        
        // zrób prefixsuma żeby je ponumerować
        thrust::inclusive_scan(parentsNumbers.begin(), parentsNumbers.end(), parentsNumbers.begin());
        //thrust::host_vector<int> parentsNumbersHost = parentsNumbers;

        // dodaj nowe nody do octree
        octree.insert(octree.end(), parentsNumbers[childrenCount-1]+1, OctreeNode()); // co tu sie odbywa?
        d_octree = thrust::raw_pointer_cast(octree.data()); // dlaczego znowu raw_cast?
        
        // czemu dodajemy tak duzo? o.o #damian
        thrust::for_each(parentsNumbers.begin(), parentsNumbers.end(), thrust::placeholders::_1 += allChildrenCount);
        
        // połącz odpowiednio dzieci
        connectChildren<<<blocks, THREADS_PER_BLOCK>>>(
            d_codes,
            d_parentsNumbers,
            d_octree,
            childrenCount,
            previousChildrenCount,
            d_sortedNodes,
            i
        );
        
        // teraz tamte co się powtarzają są dziećmi, zuniquj je
        thrust::for_each(mortonCodes.begin(), mortonCodes.end(), thrust::placeholders::_1 >>= 3);
        auto it = thrust::unique(mortonCodes.begin(), mortonCodes.end());
        childrenCount = thrust::distance(mortonCodes.begin(), it);
        previousChildrenCount = allChildrenCount;
        allChildrenCount += childrenCount; // ?? #damian
    }

    // potrzebuje jeszcze center of mass(byc moze uda sie podpiac pod , mass, totalMass #damian
    float *d_velocities = thrust::raw_pointer_cast(veloD.data());
    float *d_weights = thrust::raw_pointer_cast(weightsD.data());

    computeForces<<<blocks, THREADS_PER_BLOCK>>>(d_octree,
        d_forces, 
        d_velocities, 
        d_weights, 
        d_positions, 
        allChildrenCount, 
        N, dt);
}

void ComputationsBarnesHut::BarnesHutBridge(type &pos, int numberOfBodies, float dt) {
    thrust::device_vector<float> posD = pos;
    d_positions = thrust::raw_pointer_cast(posD.data());
    createTree(numberOfBodies);
    pos = posD;
}