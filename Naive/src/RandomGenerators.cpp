#include "RandomGenerators.h"

RandomGenerators::RandomGenerators() {

}

float RandomGenerators::getRandomFloat(float a, float b) {
    std::random_device rd1;
    std::mt19937 gen1(rd1());
    std::uniform_real_distribution<> disFloat(a, b);
    return disFloat(gen1);
}

int RandomGenerators::getRandomByte() {
    std::random_device rd2;
    std::mt19937 gen2(rd2());
    std::uniform_int_distribution<> disByte(0, 255);
    return disByte(gen2);
}
