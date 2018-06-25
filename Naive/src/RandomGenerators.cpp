#include "RandomGenerators.h"

RandomGenerators::RandomGenerators() {

}

float RandomGenerators::getRandomfloat(float a, float b) {
    std::random_device rd1;
    std::mt19937 gen1(rd1());
    std::uniform_real_distribution<> disfloat(a, b);
    return disfloat(gen1);
}

int RandomGenerators::getRandomByte() {
    std::random_device rd2;
    std::mt19937 gen2(rd2());
    std::uniform_int_distribution<> disByte(0, 255);
    return disByte(gen2);
}

int RandomGenerators::getRandomType() {
    std::random_device rd3;
    std::mt19937 gen3(rd3());
    std::uniform_int_distribution<> disInt(0, 3);
    return disInt(gen3);
}

template<typename T>
void RandomGenerators::initializeValues(T& velocities, T& weights, unsigned N) {

}

template <>
void RandomGenerators::initializeValues<std::vector<float>>(std::vector<float>& velocities, std::vector<float>& weights, unsigned N) {
    velocities.resize(3*N);
    weights.resize(N);

    for(unsigned i=0; i<N; i++)
        for(int j=0; j<3; j++)
            velocities[i*3+j] = getRandomfloat(-0.01f, 0.01f);

    int typeMass = getRandomType();
    printf("TYP %d\n", typeMass);
    switch(typeMass) {
        case 0: // full random
            for(unsigned i=0; i < N; i++)
                weights[i] = getRandomfloat(1000.0f, 100000.0f); // 10^10
            break;
        case 1: // 1/10 duze masy, reszta stosunkowo male
            for(unsigned i=0; i < (N/10); i++)
                weights[i] = getRandomfloat(1000000.0f, 1010000.0f);
            for(unsigned i=(N/10); i < N; i++)
                weights[i] = getRandomfloat(1000.0f, 2000.0f);
            break;
        case 2: // 1/20 male, 2/10 duze, 5/15 male
            for(unsigned i=0; i < (N/20); i++)
                weights[i] = getRandomfloat(1000000.0f, 1010000.0f);
            for(unsigned i=(N/20); i < 5*(N/20); i++)
                weights[i] = getRandomfloat(40000.0f, 45000.0f);
            for(unsigned i = 5*(N/20); i < N; i++)
                weights[i] = getRandomfloat(1000.0f, 2000.0f);
            break;
        default: // same male
            for(unsigned i=0; i < N; i++)
                weights[i] = getRandomfloat(1000.0f, 1100.0f); // 10^10
            break;
    }
}
// zderzenie plastyczne / sprezyste
template <>
void RandomGenerators::initializeValues<thrust::host_vector<float>>(thrust::host_vector<float>& velocities, thrust::host_vector<float>& weights, unsigned N) {
    velocities.resize(3*N);
    weights.resize(N);

    for(unsigned i=0; i<N; i++)
        for(int j=0; j<3; j++)
            velocities[i*3+j] = getRandomfloat(-0.01f, 0.01f);

    int typeMass = 0; //getRandomType();
    printf("TYP %d\n", typeMass);
    switch(typeMass) {
        case 0: // full random
            for(unsigned i=0; i < N; i++)
                weights[i] = getRandomfloat(1000.0f, 100000.0f); // 10^10
            break;
        case 1: // 1/10 duze masy, reszta stosunkowo male
            for(unsigned i=0; i < 1; i++)
                weights[i] = getRandomfloat(1000000.0f, 1010000.0f);
            for(unsigned i=1; i < N; i++)
                weights[i] = getRandomfloat(1000.0f, 2000.0f);
            break;
        case 2: // 1/20 male, 2/10 duze, 5/15 male
            for(unsigned i=0; i < (N/20); i++)
                weights[i] = getRandomfloat(1000000.0f, 1010000.0f);
            for(unsigned i=(N/20); i < 5*(N/20); i++)
                weights[i] = getRandomfloat(40000.0f, 45000.0f);
            for(unsigned i = 5*(N/20); i < N; i++)
                weights[i] = getRandomfloat(1000.0f, 2000.0f);
            break;
        default: // same male
            for(unsigned i=0; i < N; i++)
                weights[i] = getRandomfloat(1000.0f, 1100.0f); // 10^10
            break;
    }
}
