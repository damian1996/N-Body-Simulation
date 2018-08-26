#include "RandomGenerators.h"

RandomGenerators::RandomGenerators() : gen1(rd1()), gen2(rd2()) {}

float RandomGenerators::getRandomFloat(float a, float b) {
  std::uniform_real_distribution<> disfloat(a, b);
  return disfloat(gen1);
}

int RandomGenerators::getRandomColor() {
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
void RandomGenerators::initializeVelocities<std::vector<float>>(
    std::vector<float> &velocities, unsigned numberOfBodies) {
  velocities.resize(3 * numberOfBodies);
  for (unsigned i = 0; i < numberOfBodies; i++)
    for (int j = 0; j < 3; j++)
      velocities[i * 3 + j] = getRandomFloat(-0.01, 0.01);
}

template <>
void RandomGenerators::initializeVelocities<thrust::host_vector<float>>(
    thrust::host_vector<float> &velocities, unsigned numberOfBodies) {
  velocities.resize(3 * numberOfBodies);
  for (unsigned i = 0; i < numberOfBodies; i++)
    for (int j = 0; j < 3; j++)
      velocities[i * 3 + j] = getRandomFloat(-0.01f, 0.01f);
}

void RandomGenerators::initializeWeights(std::vector<float> &weights, unsigned numberOfBodies) {
  weights.resize(numberOfBodies);
  int typeMass = 3;
  unsigned divi;
  //printf("TYP %d\n", typeMass);
  switch (typeMass) {
  case 0: 
    for (unsigned i = 0; i < numberOfBodies; i++)
      weights[i] = getRandomFloat(1000.0f, 100000.0f); 
    break;
  case 1: 
    divi = static_cast<unsigned>(numberOfBodies/100);
    for (unsigned i = 0; i < divi; i++)
      weights[i] = getRandomFloat(100000.0f, 101000.0f);
    for (unsigned i = divi; i < numberOfBodies; i++)
      weights[i] = getRandomFloat(10000.0f, 20000.0f);
    break;
  case 2:
    divi = static_cast<unsigned>(numberOfBodies/20);
    for (unsigned i = 0; i < divi; i++)
      weights[i] = getRandomFloat(1000000.0f, 1010000.0f);
    for (unsigned i = divi; i < 5 * divi; i++)
      weights[i] = getRandomFloat(40000.0f, 45000.0f);
    for (unsigned i = 5 * divi; i < numberOfBodies; i++)
      weights[i] = getRandomFloat(1000.0f, 2000.0f);
    break;
  case 3:
    divi = static_cast<unsigned>(numberOfBodies/100);
    if(divi==0) divi = 1u;
    for (unsigned i = 0; i < divi; i++)
      weights[i] = getRandomFloat(1000000.0f, 1010000.0f);
    for (unsigned i = divi; (i < 5 * divi) && (i < numberOfBodies); i++)
      weights[i] = getRandomFloat(80000.0f, 95000.0f);
    for (unsigned i = 5 * divi; (i < 10 * divi) && (i < numberOfBodies); i++)
      weights[i] = getRandomFloat(60000.0f, 79000.0f);
    for (unsigned i = 10 * divi; (i < 15 * divi) && (i < numberOfBodies); i++)
      weights[i] = getRandomFloat(40000.0f, 55000.0f);
    for (unsigned i = 15 * divi; (i < 20 * divi) && (i < numberOfBodies); i++)
      weights[i] = getRandomFloat(20000.0f, 30000.0f);
    for (unsigned i = 20 * divi; i < numberOfBodies; i++)
      weights[i] = getRandomFloat(1000.0f, 3000.0f);
    break;
  default:
    for (unsigned i = 0; i < numberOfBodies; i++)
      weights[i] = getRandomFloat(1000.0f, 1100.0f); 
    break;
  }
}
