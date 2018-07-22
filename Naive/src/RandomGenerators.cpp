#include "RandomGenerators.h"

RandomGenerators::RandomGenerators() {}

double RandomGenerators::getRandomdouble(double a, double b) {
  std::random_device rd1;
  std::mt19937 gen1(rd1());
  std::uniform_real_distribution<> disdouble(a, b);
  return disdouble(gen1);
}

int RandomGenerators::getRandomByte() {
  std::random_device rd2;
  std::mt19937 gen2(rd2());
  std::uniform_int_distribution<> disByte(100, 255);
  return disByte(gen2);
}

int RandomGenerators::getRandomType() {
  std::random_device rd3;
  std::mt19937 gen3(rd3());
  std::uniform_int_distribution<> disInt(0, 3);
  return disInt(gen3);
}

template <typename T>
void RandomGenerators::initializeVelocities(T &velocities, unsigned N) {}

template <>
void RandomGenerators::initializeVelocities<std::vector<double>>(
    std::vector<double> &velocities, unsigned N) {
  velocities.resize(3 * N);
  for (unsigned i = 0; i < N; i++)
    for (int j = 0; j < 3; j++)
      velocities[i * 3 + j] = getRandomdouble(-veloRand, veloRand);
}
// zderzenie plastyczne / sprezyste
template <>
void RandomGenerators::initializeVelocities<thrust::host_vector<double>>(
    thrust::host_vector<double> &velocities, unsigned N) {
  velocities.resize(3 * N);
  for (unsigned i = 0; i < N; i++)
    for (int j = 0; j < 3; j++)
      velocities[i * 3 + j] = getRandomdouble(-veloRand, veloRand);
}

template <typename T>
void RandomGenerators::initializeWeights(T &weights, unsigned N) {}

template <>
void RandomGenerators::initializeWeights<std::vector<double>>(
    std::vector<double> &weights, unsigned N) {
  weights.resize(N);
  int typeMass = 1; // getRandomType();
  printf("TYP %d\n", typeMass);
  unsigned divi = 1;
  switch (typeMass) {
  case 0: // full random
    //weights[0] = 5000.0;
    for(unsigned i=0; i < N; i++)
        //weights[i] = getRandomdouble(100.0, 300.0); // 10^10
        weights[i] = getRandomdouble(1000.0, 100000.0); // 10^10
    break;
  case 1: // 1/10 duze masy, reszta stosunkowo male
    for (unsigned i = 0; i < (N / 10); i++)
      weights[i] = getRandomdouble(1000000.0, 1010000.0);
    for (unsigned i = (N / 10); i < N; i++)
      weights[i] = getRandomdouble(1000.0, 2000.0);
    break;
  case 2: // 1/20 male, 2/10 duze, 5/15 male
    for (unsigned i = 0; i < (N / 20); i++)
      weights[i] = getRandomdouble(1000000.0, 1010000.0);
    for (unsigned i = (N / 20); i < 5 * (N / 20); i++)
      weights[i] = getRandomdouble(40000.0, 45000.0);
    for (unsigned i = 5 * (N / 20); i < N; i++)
      weights[i] = getRandomdouble(1000.0, 2000.0);
    break;
  default: // same male
    for (unsigned i = 0; i < N; i++)
      weights[i] = getRandomdouble(1000.0, 1100.0); // 10^10
    break;
  }
}
// zderzenie plastyczne / sprezyste
template <>
void RandomGenerators::initializeWeights<thrust::host_vector<double>>(
    thrust::host_vector<double> &weights, unsigned N) {
  weights.resize(N);
  int typeMass = 0; // getRandomType();
  printf("TYPE OF DATA : %d\n", typeMass);
  switch (typeMass) {
  case 0: // full random
    for (unsigned i = 0; i < N; i++)
      weights[i] = getRandomdouble(0.5, 9.0); // 10^10
    // weights[i] = getRandomdouble(1000.0, 100000.0); // 10^10
    break;
  case 1: // 1/10 duze masy, reszta stosunkowo male
    for (unsigned i = 0; i < 1; i++)
      weights[i] = getRandomdouble(1000000.0, 1010000.0);
    for (unsigned i = 1; i < N; i++)
      weights[i] = getRandomdouble(1000.0, 2000.0);
    break;
  case 2: // 1/20 male, 2/10 duze, 5/15 male
    for (unsigned i = 0; i < (N / 20); i++)
      weights[i] = getRandomdouble(1000000.0, 1010000.0);
    for (unsigned i = (N / 20); i < 5 * (N / 20); i++)
      weights[i] = getRandomdouble(40000.0, 45000.0);
    for (unsigned i = 5 * (N / 20); i < N; i++)
      weights[i] = getRandomdouble(1000.0, 2000.0);
    break;
  default: // same male
    for (unsigned i = 0; i < N; i++)
      weights[i] = getRandomdouble(1000.0, 1100.0); // 10^10
    break;
  }
}
