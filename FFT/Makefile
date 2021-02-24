cpu:
	nvcc -lgsl -lgslcblas -lm fftCPU.cpp

gpu:
	nvcc -lcufft fftGPU.cu

clean:
	rm -f *.out *.o