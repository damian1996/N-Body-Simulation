
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

/*
    thrust::device_vector<float> centerOfMass(3*uniquePointsCount);
    float* d_centerOfMass = thrust::raw_pointer_cast(centerOfMass.data());
    thrust::fill(centerOfMass.begin(), centerOfMass.end(), 0.0);

    thrust::device_vector<float> totalMasses(uniquePointsCount);
    float* d_totalMasses = thrust::raw_pointer_cast(totalMasses.data());
    thrust::fill(totalMasses.begin(), totalMasses.end(), 0.0);
*/







/*
__device__
int delta(int i, int j, unsigned long long int* mortonCodes, int N) {
  if(i < 0 || i > N-1) return -1;
  if(j < 0 || j > N-1) return -1;
  unsigned long long a = mortonCodes[i];
  unsigned long long b = mortonCodes[j];
  return __clz(a^b)-(64-3*K);
}

__global__
void calculateRadixTree(unsigned long long int* mortonCodes, int* leftInternalChildren, 
    int* rightInternalChildren, int* nodesCountFromEdges, int N) {
  int thid = blockIdx.x*blockDim.x + threadIdx.x;
  if(thid >= N) return;
  int d = sgn(delta(thid, thid+1, mortonCodes, N) - delta(thid, thid-1, mortonCodes, N));

  int delta_min = delta(thid, thid-d, mortonCodes, N);
  int l_max = 2;
  while(delta(thid, thid*l_max*d) > delta_min) l_max *= 2;

  int l = 0;
  for(int t = l_max/2; t >= 1; t /= 2) {
    if(delta(thid, thid+(l+t)*d) > delta_min) 
      l += t;
  }
  int j = thid + l*d;

  int delta_node = delta(i, j);
  int s = 0;
  int t = (l+1)/2;
  int denom = 2;
  while(1) {
    if(delta(thid, thid+(s+t)*d) > delta_node)
      s += t;
    if(t == 1) break;
    denom *= 2;
    t = (l+denom-1)/denom;
  }
  int gamma = thid + s*d + min(d, 0);
  // TODO: co jak gamma = 0
  if(min(i, j) == gamma)
    leftInternalChildren[thid] = -gamma;
  else 
    leftInternalChildren[thid] = gamma;
  if(max(i, j) == gamma+1)
    rightInternalChildren[thid] = -(gamma+1);
  else
    rightInternalChildren[thid] = gamma+1;
  
  parent_delta = delta_node;
  left_child_delta = data();
  right_child_delta = data();
}
*/

  /*
  // 3. usuwamy duplikaty
  auto iterators = thrust::unique_by_key(mortonCodes.begin(), mortonCodes.end(), sortedNodes.begin());

  // 4. liczymy radix tree
  int uniquePointsCount = thrust::distance(mortonCodes.begin(), iterators.first);
  blocks = (uniquePointsCount+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK;
  thrust::device_vector<int> leftInternalChildren(uniquePointsCount);
  thrust::device_vector<int> rightInternalChildren(uniquePointsCount);
  int* d_leftInternalChildren = thrust::raw_pointer_cast(leftInternalChildren.data());
  int* d_rightInternalChildren = thrust::raw_pointer_cast(rightInternalChildren.data());
  calculateRadixTree<<<blocks, THREADS_PER_BLOCK>>>(
    d_mortonCodes,
    d_leftInternalChildren,
    d_rightInternalChildren,
    uniquePointsCount
  );

  // 5. liczymy ilość node-ow z octree do zaalokowania
  thrust::device_vector<int> nodesCountFromEdges(uniquePointsCount);
  int* d_nodesCountFromEdges = thrust::raw_pointer_cast(nodesCountFromEdges.data());
  calculateOctreeNodesCount<<<blocks, THREADS_PER_BLOCK>>>(
    d_mortonCodes,
    d_leftInternalChildren,
    d_rightInternalChildren,
    d_nodesCountFromEdges,
    uniquePointsCount-1,
  );

  thrust::device_vector<int> inclusiveSum(uniquePointsCount);
  thrust::device_vector<int> exclusiveSum(uniquePointsCount);

  thrust::inclusive_scan(nodesCountFromEdges.begin(), nodesCountFromEdges.end(), inclusiveSum.begin());
  // todo: can be done better - just substract the values
  thrust::exclusive_scan(nodesCountFromEdges.begin(), nodesCountFromEdges.end(), exclusiveSum.begin());

  int octreeInternalNodesCount = 1+inclusiveSum.back();
  int octreeNodesCount = octreeInternalNodesCount+uniquePointsCount;
  thrust::device_vector<OctreeNode> octree(octreeNodesCount);
  OctreeNode* d_octree = thrust::raw_pointer_cast(octree.data());

  // 6. laczymy octree nodes
  blocks = (octreeInternalNodesCount+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK;
  int* d_inclusiveSum = thrust::raw_pointer_cast(inclusiveSum.data());
  int* d_exclusiveSum = thrust::raw_pointer_cast(exclusiveSum.data());
  connectOctreeNodes<<<blocks, THREADS_PER_BLOCK>>>(
    d_mortonCodes,
    d_leftInternalChildren,
    d_rightInternalChildren,
    d_inclusiveSum,
    d_exclusiveSum,
    uniquePointsCount,
    octreeInternalNodesCount,
  );
  // TODO: return the tree! xd
  */