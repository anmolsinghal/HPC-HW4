jacobi : driver.cu jacobi.cu jacobi.cuh
	nvcc -std=c++11 driver.cu jacobi.cu -o jacobi2D-cuda
matvecmult: q1_final.cu
	nvcc -std=c++11 q1_final.cu -o matvecmult
all : jacobi matvecmult
clean:
	rm -f *.out
	rm jacobi2D-cuda
	rm matvecmult
