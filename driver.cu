#include <iostream>
#include "jacobi.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <string>

void swap(double **r, double **s)
{
    double *pSwap = *r;
    *r = *s;
    *s = pSwap;
}

int main(int argc, char * argv[]) 
{
    int N;
    if(argc < 2) N = 100;
    else N = std::stoi(argv[1]);

    int max_iter;
    if(argc < 3) max_iter = 1000; 
    else max_iter = std::stoi(argv[2]);

    int dimensions[2] = {N, N}, 
    u_mem_required = dimensions[0] * dimensions[1] * sizeof(double);
    

    double * start_u, * f, * u, * u_new, * f_device;
   
    const dim3 blockSize( 10 , 10), gridSize( dimensions[0] / 10, dimensions[1] / 10 );

    std::cout << "Initializing u to 0 and f to 1" << std::endl;
    
    start_u = new double[dimensions[0] * dimensions[1]];
    f = new double[dimensions[0] * dimensions[1]];
    for(int i = 0; i < dimensions[0]; i++) {
        int offset = i * dimensions[1];
        for(int j = 0; j < dimensions[1]; j++) {
            start_u[offset + j] = 0;
            f[offset + j] = 1;
        }
    }

    printf("Initial residual = %e\n",calculate_residual(start_u, f, dimensions));

    std::cout << "Copying to Device" << std::endl;
    try 
    {
        copyToDevice(start_u, f, dimensions, &u, &u_new, &f_device);
    }
    catch( ... )
    {
        printf("Exception happened while copying to device\n");
    }

    double h = 1.0 / (N + 1);
    double h2 = h*h;
    printf("Perform Jacobi iterations\n");
    for( int i = 0; i < max_iter; i++)
    {
        doJacobiIteration<<< gridSize, blockSize >>>(dimensions[0], dimensions[1], u, u_new, f_device, h2);
        cudaDeviceSynchronize();
        if(cudaGetLastError() != cudaSuccess)
        {
            prinft("Error Launching Kerne\n");
            return 1;
        }
        swap(&u, &u_new);
        
    }

    
    if(cudaMemcpy( start_u, u, u_mem_required, cudaMemcpyDeviceToHost ) != cudaSuccess) 
    {
        printf("There was a problem retrieving the result from the device\n");
        return 1;    
    }
    
    printf("Final Jacobi residual = %e\n",calculate_residual(start_u, f, dimensions)) ;
    cudaFree(u);
    cudaFree(u_new);
    delete [] start_u;
    delete [] f;

    return 0;
}
