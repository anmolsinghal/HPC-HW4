#include<stdio.h>
#include<cuda.h>

#define BLOCKSIZE 16
#define EPS 1.0e-15

cudaDeviceProp deviceProp;	


double *host_matrix,*host_vector,*host_result,*cpu_result;
double *device_matrix,*device_vector,*device_result;
int     v_length ,mat_row_size , mat_col_size;
int     size = BLOCKSIZE;

double calculate_gbs(float &Tsec)
{
        float bw=(1.0e-9 * (( size*size + size )/Tsec));
	return bw;
}

void cpu_multiply()
{
	cpu_result = (double *)malloc(mat_row_size*sizeof(double));
	if(cpu_result==NULL)
                {
                        printf("Not enough memory");
                        exit(-1);
                }

	int i,j;
	for(i=0;i<mat_row_size;i++)
	{cpu_result[i]=0;
	for(j=0;j<mat_col_size;j++)
	cpu_result[i]+=host_matrix[i*v_length+j]*host_vector[j];
	}
}

void device_free(double * arr[],int len)
{
        for(int i=0;i<len;i++)
                cudaFree(arr[i]);
        
}

/* function to calculate relative error*/
void relative_error(double* device,double* host,int size)
{
        double relativeError=0.0,maxError=0.0;
        int flag=0;
        int i;

        for( i = 0; i < size; ++i) 
        {
               
                relativeError = fabs((host[i] - device[i]) )/ max(fabs(host[i]), fabs(device[i]));
                
                if (relativeError > EPS && relativeError != 0.0e+00 )
                {       
                        maxError = max(maxError, relativeError);
                        flag = 1;                        
                }

        }
        if( flag == 1)
        {
                printf(" \n Verification failed with error %e on machine with precision %e", maxError, EPS);
        }
        
}

void fill_matrix(double* vec,int size)
{
        int ind;
        for(ind=0;ind<size;ind++)
                vec[ind]=drand48();
}


__global__ void MatVectMultiplication(double *device_matrix, double *device_vector,int mat_row_size, int v_length,double *device_result)
  {
        int tidx = blockIdx.x*blockDim.x + threadIdx.x;
        int tidy = blockIdx.y*blockDim.y + threadIdx.y;
        int tindex=tidx+gridDim.x*BLOCKSIZE*tidy;


        if(tindex<mat_row_size)
	{
                int i;int m=tindex*v_length;
                device_result[tindex]=0.00;
                for(i=0;i<v_length;i++)
                device_result[tindex]+=device_matrix[m+i]*device_vector[i];
	}

     __syncthreads();

  }//end of MatVect device function



void MatVectMul()
{
        int max=BLOCKSIZE*BLOCKSIZE;
        int BlocksPerGrid= mat_row_size/max+1;
        dim3 dimBlock(BLOCKSIZE,BLOCKSIZE);
        if(mat_row_size%max==0)BlocksPerGrid--;
        dim3 dimGrid(1,BlocksPerGrid);
        
        MatVectMultiplication<<<dimGrid,dimBlock>>>(device_matrix,device_vector,mat_row_size,v_length,device_result);

}


double sim()
{
       	v_length = mat_col_size = mat_row_size = size;
       	
	float elapsedTime,Tsec;
	cudaEvent_t start,stop;


	host_matrix =new double[mat_row_size*mat_col_size];
	host_vector = new double[v_length];
	host_result = new double[mat_row_size];

	
        if(host_matrix==NULL || host_vector == NULL || host_result == NULL)
        {
                printf("Not enough memory\n");
                exit(-1);
        }

	fill_matrix(host_matrix,mat_row_size*mat_col_size);
	fill_matrix(host_vector,v_length);

 	
        cudaEventCreate (&start);
        cudaEventCreate (&stop);

	cudaMalloc( (void**)&device_matrix, mat_row_size*mat_col_size* sizeof(double));
	cudaMalloc( (void**)&device_vector, v_length* sizeof(double));
	cudaMalloc( (void**)&device_result, mat_row_size* sizeof(double));

	cudaMemcpy((void*)device_matrix, (void*)host_matrix, mat_row_size*mat_col_size*sizeof(double) ,cudaMemcpyHostToDevice);
	cudaMemcpy((void*)device_vector, (void*)host_vector,v_length*sizeof(double),cudaMemcpyHostToDevice);

	cudaEventRecord (start, 0);
	
	MatVectMul();
	
	cudaEventRecord (stop, 0);
	cudaEventSynchronize (stop);
	cudaEventElapsedTime ( &elapsedTime, start, stop);

	Tsec= 1.0e-3*elapsedTime;
 	
        double ret = calculate_gbs(Tsec);
	
	
  	cudaMemcpy((void*)host_result, (void*)device_result,mat_row_size*sizeof(double),cudaMemcpyDeviceToHost);

	cpu_multiply();
  	relative_error(cpu_result,host_result,size);
   	/*free the memory from GPU */
	double *array[3];
        array[0]=device_matrix;
        array[1]=device_vector;
        array[2]=device_result;
        device_free(array,3);

	//free host memory----------
        free(host_matrix);
        free(host_vector);
        free(host_result);
        free(cpu_result);

	return ret;
}// end of main


int main()
{       
       
        cudaSetDevice(0);
      
        int device;
        // Current Device Detection
        cudaGetDevice(&device);
        cudaGetDeviceProperties(&deviceProp,device);
        printf("Using device %d: %s \n", device, deviceProp.name);

        printf("Size \t \t Bandwith\n");
        for(size = 16 ;size <= 8192;size *=2)
        {
                double gbs = sim();
                printf("%d \t \t %f\n", size, gbs);
        }


}