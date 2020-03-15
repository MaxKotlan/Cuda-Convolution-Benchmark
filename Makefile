all:
	nvcc -g -o Convolution Main.cu Test.cu Benchmark.cu
