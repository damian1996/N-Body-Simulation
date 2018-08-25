#include "StepNaiveCUDA.h"

StepNaiveCuda::StepNaiveCuda(std::vector<float> masses, unsigned numberOfBodies) {
  this->numberOfBodies = numberOfBodies;
  weights.resize(numberOfBodies);
  for (unsigned i = 0; i < numberOfBodies; i++) {
    weights[i] = masses[i];
  }
  randomGenerator = new RandomGenerators();
  randomGenerator->initializeVelocities<thrust::host_vector<float>>(velocities, numberOfBodies);
  c = new Computations(velocities, weights);
}

StepNaiveCuda::~StepNaiveCuda() {
  delete c;
  weights.clear();
  velocities.clear();
  delete randomGenerator;
}

void StepNaiveCuda::compute(tf3 &positions, float dt) {
  c->NaiveSimBridgeThrust(positions, numberOfBodies, dt);
}

// https://www.bu.edu/pasi/files/2011/07/Lecture6.pdf
// https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html
// https://stackoverflow.com/questions/4176762/passing-structs-to-cuda-kernels
// https://codeyarns.com/2011/02/16/cuda-dim3/
// http://developer.download.nvidia.com/CUDA/training/introductiontothrust.pdf
// https://groups.google.com/forum/#!topic/thrust-users/4EaWLGeJOO8
// https://github.com/thrust/thrust/blob/master/examples/cuda/unwrap_pointer.cu
// thrust::copy(weightsD.begin(), weightsD.end(),
// std::ostream_iterator<float>(std::cout, " "));
