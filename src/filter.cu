#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "helper_cuda.h"

// Global function means it will be executed on the device (GPU)
__global__ void trim(
    float * const __restrict__ inx,
    float * const __restrict__ iny,
    float * const __restrict__ inz,
    int const size)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;

  if( inz[index] < 0.3 && index < size){
    inx[index] = 0.0;
    iny[index] = 0.0;
    inz[index] = 0.0;
  }
}

// Global function means it will be executed on the device (GPU)
__global__ void distance(
    float * const __restrict__ inx,
    float * const __restrict__ iny,
    float * const __restrict__ inz,
    int * const __restrict__ indexes,
    float const origin_x,
    float const origin_y,
    float const origin_z,
    float const mark_range,
    int const size)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  float distance = inx[index];

  if (index < size){
    // distance =  (inx[index] - origin_x) * (inx[index] - origin_x) +
    //             (iny[index] - origin_y) * (iny[index] - origin_y) +
    //             (inz[index] - origin_z) * (inz[index] - origin_z);
    //
    if (distance > mark_range || distance < 0.0001){
      indexes[index] = 1;
    } else {
      indexes[index] = 0;
    }
  }
}

template<typename T>
T div_round_up(T a, T b) {
  return (a + b - 1) / b;
}

float *filter(
    float * inx,
    float * iny,
    float * inz,
    size_t size, size_t threads)
{
  printf("Filter! %ld\n", size);

  trim<<<div_round_up(size,threads),threads>>>(inx, iny, inz, size);
  getLastCudaError("trim() kernel failed");

  // Wait for the GPU to finish
  cudaDeviceSynchronize();

  return inx;
}

float *compute_distance(
    float * inx,
    float * iny,
    float * inz,
    int * index_array,
    float origin_x,
    float origin_y,
    float origin_z,
    float mark_range,
    size_t size, size_t threads)
{
  printf("Compute Distance! %ld\n", size);

  distance<<<div_round_up(size,threads),threads>>>(inx, iny, inz, index_array, \
      origin_x, origin_y, origin_z, mark_range, size);
  getLastCudaError("distance() kernel failed");

  // Wait for the GPU to finish
  cudaDeviceSynchronize();

  return inx;
}



