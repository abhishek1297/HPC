#include <cstdio>
#include <iostream>
#include <algorithm>
#include <cufft.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>

#define N 16
typedef float2 Complex;

int main () {

    const int MEM_SIZE = sizeof(Complex) * N;
    //----------------Initialize-----------------
    Complex *h_data = reinterpret_cast<Complex *>(malloc(MEM_SIZE)); //float2
    std::fill(h_data, h_data + N, make_float2(0.0, 0.0));
    for (int i = 0; i < 5; ++i) {
        h_data[i].x = h_data[N - i].x = 1.0;
    }

    printf("Original signal:\n");
    for (int i = 0; i < N; ++i) {
        printf("%d %e %e\n", i, h_data[i].x, h_data[i].y);
    }

    Complex *d_data;
    cudaMalloc(reinterpret_cast<void **>(&d_data), MEM_SIZE);
    cudaMemcpy(d_data, h_data, MEM_SIZE, cudaMemcpyHostToDevice);

    cufftHandle plan;
    cufftPlan1d(&plan, N, CUFFT_C2C, 1);
    //----------------Perform FFT----------------------
    cufftExecC2C(plan, reinterpret_cast<cufftComplex *>(d_data),
    reinterpret_cast<cufftComplex *>(d_data),
    CUFFT_FORWARD);
    cudaMemcpy(h_data, d_data, MEM_SIZE, cudaMemcpyDeviceToHost);
    printf("\n\nAfter GPU FFT:\n");
    for (int i = 0; i < N; ++i) {
        printf("%d %e %e\n", i, h_data[i].x/sqrt(N), h_data[i].y/sqrt(N));
    }
    //----------------Perform Inverse FFT--------------
    cufftExecC2C(plan, reinterpret_cast<cufftComplex *>(d_data),
    reinterpret_cast<cufftComplex *>(d_data),
    CUFFT_INVERSE);

    cudaMemcpy(h_data, d_data, MEM_SIZE, cudaMemcpyDeviceToHost);
    printf("\n\nAfter GPU Inverse FFT:\n");
    for (int i = 0; i < N; ++i) {
        printf("%d %e %e\n", i, h_data[i].x, h_data[i].y);
    }
    cudaFree(d_data);
    free(h_data);

    return 0;
}