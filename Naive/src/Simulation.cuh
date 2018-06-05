#ifndef SIMULATIONCUDA_H
#define SIMULATIONCUDA_H

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "render.h"

typedef thrust::host_vector<float> type;

template <typename T>
struct KernelArray
{
    T*  arr;
    int _size;
    KernelArray(int N) {
        arr = (T*)malloc(N*sizeof(T));
    }
    KernelArray(thrust::device_vector<T>& dVec, int N) {
        arr = (T*)malloc(N*sizeof(T));
        arr = thrust::raw_pointer_cast(dVec.data());
        _size  = (int) N;
    }
    ~KernelArray() {
        free(arr);
    }
};

// Function to convert device_vector to structure

template <typename T>
KernelArray<T>* cTK(thrust::device_vector<T>& dVec, int N)
{
    KernelArray<T>* kArray = new KernelArray<T>(N);
    //kArray->arr = thrust::raw_pointer_cast(&dVec[0]);
    kArray->arr = thrust::raw_pointer_cast(dVec.data());
    kArray->_size  = (int) N;

    return kArray;
}

void NaiveSimBridge(Scene* painter, type& pos, type& velocities, type& weights, int N);
void NaiveSimBridgeThrust(Scene* painter, type& pos, type& velocities, type& weights, int N);
#endif
