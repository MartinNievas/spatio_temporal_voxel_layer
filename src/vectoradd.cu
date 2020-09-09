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
    inz[index] = 0.3;
  }
}

void random_ints(int *i, int size)
{
  for(int k=0; k<size; k++)
  {
    i[k]=rand()%50;
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
  printf("Filter! %d\n", size);

  trim<<<div_round_up(size,threads),threads>>>(inx, iny, inz, size);
  getLastCudaError("trim() kernel failed");

  // Wait for the GPU to finish
  cudaDeviceSynchronize();

  return inx;
}


