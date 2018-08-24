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

//konstruktor
struct OctreeNode {
  int children[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
  int position = -1;
  float totalMass = 0;
  float centerX = 0.0;
  float centerY = 0.0;
  float centerZ = 0.0;
  //float boundaries[6];
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
    int N, int previousChildrenCount, int* sortedNodes, float* positions, float* weights, int level) {
  int thid = blockIdx.x*blockDim.x + threadIdx.x;
  if(thid >= N) return;
  // dziecko łączy się z parentsNumbers[thid] pod wskaźnikiem zależnym od bitów
  unsigned long long int childNumber = mortonCodes[thid]&0x7; // 7 = 111 binarnie
  octree[parentsNumbers[thid]].children[childNumber] = thid+previousChildrenCount;
  octree[parentsNumbers[thid]].position = -1;
  octree[thid+previousChildrenCount].position = level == 0 ? sortedNodes[thid] : -1;

  if(!level) {
      // setup leaves, will be bad by removing of duplicates
      octree[sortedNodes[thid]].totalMass = weights[thid];
      octree[sortedNodes[thid]].centerX = (weights[thid] * positions[3*thid]) / weights[thid];
      octree[sortedNodes[thid]].centerY = (weights[thid] * positions[3*thid]) / weights[thid];
      octree[sortedNodes[thid]].centerZ = (weights[thid] * positions[3*thid]) / weights[thid];
  }
  else {
      octree[parentsNumbers[thid]].totalMass += octree[childNumber].totalMass;
      octree[parentsNumbers[thid]].centerX += octree[childNumber].centerX;
      octree[parentsNumbers[thid]].centerY += octree[childNumber].centerY;
      octree[parentsNumbers[thid]].centerZ += octree[childNumber].centerZ;
  }
}

__global__
void computeForces(OctreeNode* octree, float* velocities, float* weights, 
    float* pos, float* mins, float* maxs, int AllNodes, int N, float dt) 
{
    int thid = blockIdx.x*blockDim.x + threadIdx.x;
    if(thid >= N) return;

    int multipliers[8][3] = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}, {0, 0, 1}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1}};
    float forces[3] = {0.0, 0.0, 0.0}; 
    // stos par (element, kolejny child elementu)
    unsigned int stack[40]; // maybe long long?
    unsigned int child[40];
    float b[6*40];
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
        
       // printf("%d %d\n", idx, nextChild);
        if(octree[idx].position == -1)
        {
            //printf("%d %d\n", idx, nextChild);
            double d = EPS*EPS;
            d += (octree[idx].centerX - pos[3*thid])*(octree[idx].centerX - pos[3*thid]);
            d += (octree[idx].centerY - pos[3*thid + 1])*(octree[idx].centerY - pos[3*thid + 1]);
            d += (octree[idx].centerZ - pos[3*thid + 2])*(octree[idx].centerZ - pos[3*thid + 2]);
            d = d * sqrt(d);
            double s = b[6*prevTop + 1] - b[6*prevTop]; //boundaries[1] - boundaries[0];
            bool isFarAway = (s/d < theta);
            
            if(isFarAway)
            {
                //printf("is far away 1\n");
                double distX = octree[idx].centerX - pos[0];
                double distY = octree[idx].centerY - pos[1];
                double distZ = octree[idx].centerZ - pos[2];
                double dist = (distX * distX + distY * distY + distZ * distZ) + EPS * EPS;
                dist = dist * sqrt(dist);
                float F = G * (octree[idx].totalMass * weights[thid]);
                forces[0] += F * distX / dist;
                forces[1] += F * distY / dist;
                forces[2] += F * distZ / dist;
                //printf("is far away 2\n");
            }
            else
            { 
                // wrzuca odwrotnie, zmienic zwrot i punkt zaczepienia petli
                //rekurencja pomijajac ratio
                if(nextChild==numberOfChilds) {
                    continue;
                }
                stack[++top] = idx;
                child[top] = nextChild + 1;
                //if(thid >=8 && thid <=10)
                //printf("Pierwsze para : %d %d \n", stack[top], child[top]);
                for(int j=0; j<6; j++)
                    stack[6*top + j] = stack[6*prevTop + j];

                // if(octree[idx].position >= 0) // tutaj niespelnione trywialnie
                // mozliwe, ze musze tez sprawdzac pierwszy case, czy sasiad nie jest -1.. a tak, to problem z glowy ;)
                //if(octree[idx].children[nextChild] == -1) 
                //    continue;

                stack[++top] = octree[idx].children[nextChild];
                child[top] = 0;
                //if(thid >=8 && thid <=10)
                //printf("Druga para : %d %d \n", stack[top], child[top]);
                for(int i=0; i<3; i++) {
                    if(multipliers[nextChild][i]) {
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
            if(thid == octree[idx].position) // tutaj duplikaty msuza byc zachowane ://
                continue;
            
            float distX = pos[3*idx] - pos[3*thid];
            float distY = pos[3*idx + 1] - pos[3*thid + 1];
            float distZ = pos[3*idx + 2] - pos[3*thid + 2];
            float dist = (distX * distX + distY * distY + distZ * distZ) + EPS * EPS;
            dist = dist * sqrt(dist);
            float F = G * (weights[octree[idx].position] * weights[thid]);
            forces[0] += F * distX / dist; // force = G(m1*m2)/r^2
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

void ComputationsBarnesHut::createTree(int numberOfBodies, float dt) {
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
    // printf("UNIQUE POINTS COUNT %d\n", uniquePointsCount);
    blocks = (uniquePointsCount+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK;

    thrust::device_vector<OctreeNode> octree(uniquePointsCount);
    OctreeNode* d_octree = thrust::raw_pointer_cast(octree.data());
	
    // 6. laczymy octree nodes
    thrust::device_vector<int> parentsNumbers(uniquePointsCount); // moze za malo elementow? bo pozniej tutaj nie robimy inserta...
    int* d_parentsNumbers = thrust::raw_pointer_cast(parentsNumbers.data());
    int childrenCount = uniquePointsCount;
    int allChildrenCount = uniquePointsCount;
    int previousChildrenCount = 0;
    
    for(int i = 0; i < K; ++i) {
        printf("Aha... %d %d\n", i, allChildrenCount);
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
        //printf("Punkt kontrolny 111\n");
        // dodaj nowe nody do octree
        octree.insert(octree.end(), parentsNumbers[childrenCount-1]+1, OctreeNode()); // co tu sie odbywa?
        d_octree = thrust::raw_pointer_cast(octree.data()); // dlaczego znowu raw_cast?
        
        thrust::for_each(parentsNumbers.begin(), parentsNumbers.end(), thrust::placeholders::_1 += allChildrenCount);

       // printf("Punkt kontrolny 222\n");
        // połącz odpowiednio dzieci
        connectChildren<<<blocks, THREADS_PER_BLOCK>>>(
            d_codes,
            d_parentsNumbers,
            d_octree,
            childrenCount,
            previousChildrenCount,
            d_sortedNodes,
            d_positions,
            d_weights,
            i
        );
       // printf("Punkt kontrolny 333\n");

        // teraz tamte co się powtarzają są dziećmi, zuniquj je
        thrust::for_each(mortonCodes.begin(), mortonCodes.end(), thrust::placeholders::_1 >>= 3);
        
        thrust::host_vector<int> tmpCodes = mortonCodes;
        //float* testCodes = thrust::raw_pointer_cast(veloD.data());
        if(!std::is_sorted(tmpCodes.begin(), tmpCodes.end())) {
            printf("No nie jest posortowany :/ \n");
            thrust::sort(mortonCodes.begin(), mortonCodes.end());
        }
        
        auto it = thrust::unique(mortonCodes.begin(), mortonCodes.end());
        childrenCount = thrust::distance(mortonCodes.begin(), it);
        //std::printf("After unique...%d\n", childrenCount);
        previousChildrenCount = allChildrenCount;
        allChildrenCount += childrenCount;
    }
    //printf("Ze co? %d\n", allChildrenCount);
    float *d_velocities = thrust::raw_pointer_cast(veloD.data());
    float *d_weights = thrust::raw_pointer_cast(weightsD.data());
    
    /*
    computeForces<<<blocks, THREADS_PER_BLOCK>>>(d_octree,
        d_velocities, 
        d_weights, 
        d_positions, 
        d_mins, d_maxs,
        allChildrenCount, 
        numberOfBodies, dt); 
    */
}

void ComputationsBarnesHut::BarnesHutBridge(type &pos, int numberOfBodies, float dt) {
    thrust::device_vector<float> posD = pos;
    d_positions = thrust::raw_pointer_cast(posD.data());
    createTree(numberOfBodies, dt);
    pos = posD;
}