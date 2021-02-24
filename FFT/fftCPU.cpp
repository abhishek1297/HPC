#include <iostream>
#include <cmath>
#include <cstdlib>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_fft_complex.h>

#define N 16
typedef double Complex;

int main () {


    Complex *data = reinterpret_cast<Complex *>(malloc(sizeof(Complex) * 2 * N));

    //-------------Initialize Data---------------
    std::fill(data, data + (2 * N), 0.0);
    for (int i=0; i<5; ++i) {
        data[2*i] = data[2*(N-i)] = 1.0;
    }
    for (int i=0; i<N; ++i) {
		printf("%d %e %e\n", i, data[2*i], data[2*i + 1]);
	}

    //--------------Perform FFT------------------
    gsl_fft_complex_radix2_forward (data, 1, N);

	printf("\n\nAfter GSL FFT:\n");
	for (int i=0; i<N; ++i) {
		printf("%d %e %e\n", i, data[2*i]/sqrt(N), data[2*i + 1]/sqrt(N));
	}
    //--------------Perform Inverse FFT------------------
	gsl_fft_complex_radix2_inverse(data, 1, N);
	
    printf("\n\nAfter GSL Inverse FFT:\n");
	for (int i=0; i<N; ++i) {
		printf("%d %e %e\n", i, data[2*i], data[2*i + 1]);
	}
 
    free(data);

    return 0;
}