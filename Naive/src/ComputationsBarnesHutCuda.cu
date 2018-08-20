#include "ComputationsBarnesHutCuda.h"
#include <thrust/sort.h>
#include <bitset>
#include <string>

const float G = 6.674 * (1e-11);
const float EPS = 0.01f;

const int THREADS_PER_BLOCK = 1024;
const int K = 20;

struct OctreeNode {
  int children[8];
  int position;
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

// napisac to lepiej
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
void computeForces(float* forces, float* velocities, float* weights, int N) {
    // informacje czy jest punktem/czy ma dzieci (jeden z trzech stanow)
    // pozycje, centerOfMass, mass, totalMass, 
    unsigned long long int stack[64];
    //stack[0] = root;

    /*
     if(r->isPoint() && !r->wasInitialized())
    {
        if(r->getIndex() == i) return; // ten sam Node

        // Jesli node jest zewnetrzny, to policz sile ktora wywiera ten node na obecnie rozwazane cialo
        double distX = r->getSelectedPosition(0) - pos[0];
        double distY = r->getSelectedPosition(1) - pos[1];
        double distZ = r->getSelectedPosition(2) - pos[2];
        double dist = (distX * distX + distY * distY + distZ * distZ) + EPS * EPS;
        dist = dist * sqrt(dist);
        // jak sprawdzic czy to te same bodies? Musze zachowac i != j
        float F = G * (r->getMass() * weights[i]);
        forces[i * 3] += F * distX / dist; // force = G(m1*m2)/r^2
        forces[i * 3 + 1] += F * distY / dist;
        forces[i * 3 + 2] += F * distZ / dist;
    }
    else if(!r->isPoint() && r->wasInitialized())
    {
        if(r->isInQuad(pos[0], pos[1], pos[2]))
        {
            //rekurencja pomijajac ratio
            for(auto* child : r->getQuads())
            {
                computeForceForBody(child, pos, i);
            }
            return;
        }

        std::array<double, 6>& boundaries = r->getBoundaries();

        double d = distanceBetweenTwoNodes(pos[0], pos[1], pos[2],
            r->getSelectedCenterOfMass(0),
            r->getSelectedCenterOfMass(0),
            r->getSelectedCenterOfMass(0));
        double s = boundaries[1] - boundaries[0];

        bool isFarAway = (s/d < theta);
        if(isFarAway)
        {
            double distX = r->getSelectedCenterOfMass(0) - pos[0];
            double distY = r->getSelectedCenterOfMass(1) - pos[1];
            double distZ = r->getSelectedCenterOfMass(2)- pos[2];
            double dist = (distX * distX + distY * distY + distZ * distZ) + EPS * EPS;
            dist = dist * sqrt(dist);
            // jak sprawdzic czy to te same bodies? Musze zachowac i != j
            float F = G * (r->getTotalMass() * weights[i]);
            forces[i * 3] += F * distX / dist; // force = G(m1*m2)/r^2
            forces[i * 3 + 1] += F * distY / dist;
            forces[i * 3 + 2] += F * distZ / dist;
        }
        else
        {
            for(auto* child : r->getQuads())
            {
                computeForceForBody(child, pos, i);
            }
        }
    }
    */
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
}

void ComputationsBarnesHut::createTree(int numberOfBodies, type &pos) {
    int blocks = (numberOfBodies+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK;
    // 0. liczymy boundaries

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
    
    // moje dodatki
    thrust::device_vector<float> centerOfMass(3*uniquePointsCount);
    float* d_centerOfMass = thrust::raw_pointer_cast(centerOfMass.data());
    thrust::fill(centerOfMass.begin(), totalMasses.end(), 0.0);

    thrust::device_vector<float> totalMasses(uniquePointsCount);
    float* d_totalMasses = thrust::raw_pointer_cast(totalMasses.data());
    thrust::fill(totalMasses.begin(), totalMasses.end(), 0.0);
    // moje dodatki

    thrust::device_vector<OctreeNode> octree(uniquePointsCount);
    OctreeNode* d_octree = thrust::raw_pointer_cast(octree.data());

    // 6. laczymy octree nodes
    thrust::device_vector<int> parentsNumbers(uniquePointsCount);
    int* d_parentsNumbers = thrust::raw_pointer_cast(parentsNumbers.data());
    int childrenCount = uniquePointsCount;
    int allChildrenCount = uniquePointsCount;
    int previousChildrenCount = 0;
    
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
        thrust::host_vector<int> parentsNumbersHost = parentsNumbers;

        // dodaj nowe nody do octree
        octree.insert(octree.end(), parentsNumbers[childrenCount-1]+1, OctreeNode()); // co tu sie odbywa?
        d_octree = thrust::raw_pointer_cast(octree.data()); // dlaczego znowu raw_cast?
        
        // czemu dodajemy tak duzo? o.o
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
        allChildrenCount += childrenCount; // ??
    }

    // potrzebuje jeszcze center of mass(byc moze uda sie podpiac pod , mass, totalMass
    float *d_velocities = thrust::raw_pointer_cast(veloD.data());
    float *d_weights = thrust::raw_pointer_cast(weightsD.data());
    // tutaj liczenie sily i nowych pozycji
    //computeForces(d_forces, d_weights, d_velocities, N);
}

void ComputationsBarnesHut::BarnesHutBridge(type &pos, int numberOfBodies, float dt) {
  thrust::device_vector<float> posD = pos;
  d_positions = thrust::raw_pointer_cast(posD.data());
  createTree(numberOfBodies, pos);
  pos = posD;
}


//int* d_exclusiveSum = thrust::raw_pointer_cast(exclusiveSum.data());
  /*
  // każdy wewnętrzny r-node ma przypisane jakieś (lub żadne) o-node
  // dla każdego wewnętrznego r-node:
  blocks = (N-1+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK;

  getPrefixes<<<blocks, THREADS_PER_BLOCK>>>(
    d_codes,
    d_leftInternalChildren,
    d_rightInternalChildren,
    d_inclusiveSum,
    d_exclusiveSum,
    d_prefixes,
    d_prefixesBitsCount,
    N-1,
  );

  // TODO: sortujemy te prefixy
  // najpierw po prefixach, potem po blankach (najpierw więcej potem mniej) i jeszcze jest klucz
  thrust::zip_iterator<thrust::pair<
    thrust::device_vector<long long int>::iterator,
    thrust::device_vector<int>::iterator
  >> zip_iter(thrust::make_pair(prefixes.begin(), prefixesBitsCount.begin()));
  // TODO: czy na pewno sortedNodes?, nie tu trzeba nową tablice :v
  thrust::sort_by_key(zip_iter.begin(), zip_iter.end(), sortedNodes.begin());
  
  // dla każdego wewnętrznego o-node szukamy jego rodzica
  getParents<<<blocks, THREADS_PER_BLOCK>>>(
    d_codes,
    d_
  );
  
  // dla każdego liścia o-node szukamy jego rodzica
  */

  // TODO: return the tree! xd