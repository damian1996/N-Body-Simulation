#ifndef RANDOMGENERATORS_H
#define RANDOMGENERATORS_H

#include <random>
#include <thrust/host_vector.h>
#include <vector>

class RandomGenerators {
public:
  const double veloRand = 0.1;
  RandomGenerators();
  double getRandomdouble(double a, double b);
  // double randPosition(double a, double b);
  int getRandomByte();
  int getRandomType();
  template <typename T> void initializeVelocities(T &velocities, unsigned N);

  template <typename T> void initializeWeights(T &weights, unsigned N);
};

#endif
