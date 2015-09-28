#include <emmintrin.h>
#include <omp.h>
#define KERNX 3 //this is the x-size of the kernel. It will always be odd.
#define KERNY 3 //this is the y-size of the kernel. It will always be odd.
/* OPENMP settings. */
#define NUM_THREADS 6  //6 is the best for 330 soda, 2 for second floor soda


int conv2D(float* in, float* out, int data_size_X, int data_size_Y,
                    float* kernel)
{
	int kern_cent_X = (KERNX - 1)/2;
	int kern_cent_Y = (KERNY - 1)/2;
	int newKERNX = KERNX+(4-KERNX%4);   
	float zeropaddedKernel[newKERNX*KERNY]; 
		
	int newX = data_size_X+((KERNX/2)*2); 
	int newY = data_size_Y+((KERNX/2)*2);
	float tmp;
	
	int size = data_size_X * data_size_Y;
	float *copy_out = (float *) malloc(sizeof(float) * size); 
	
	omp_set_num_threads(NUM_THREADS);
    #pragma omp parallel firstprivate(kern_cent_X, kern_cent_Y, newKERNX, zeropaddedKernel, newX, newY, tmp) 
	{ 
		#pragma omp for schedule(dynamic, NUM_THREADS) nowait
		for(int z = 0; z < newKERNX*KERNY; z++)     
			zeropaddedKernel[z] = 0;
		
		for(int q = -kern_cent_Y; q <= kern_cent_Y; q++) {
			for(int w = -kern_cent_X; w <= kern_cent_X; w++) {
				zeropaddedKernel[(w+kern_cent_X)+(q+kern_cent_Y)*newKERNX] = kernel[(kern_cent_X-w)+(kern_cent_Y-q)*KERNX];   
			}
		}

		newX += (4-(newX%4));	
		float zeropadded[newX*newY]; 
		
		//#pragma omp unroll
		for(int z = 0; z < newX*newY; z++)     
			zeropadded[z] = 0;       	

		for(int b = 0; b < data_size_Y; b++){
			//#pragma omp unroll private(b, newX, data_size_X, data_size_Y)
			for(int a = 0; a < data_size_X; a++){
				zeropadded[(a+(KERNX/2))+(b+(KERNX/2))*newX] = in[a + b*data_size_X];
			}
		}
		    				
		#pragma omp for private(tmp) schedule(dynamic, NUM_THREADS) nowait
		for(int y = 0; y < data_size_Y ; y++){ 
			#pragma omp unroll
			for(int x = 0; x < data_size_X ; x++){ 
				tmp = 0.0;
				#pragma omp unroll
				for(int j = 0; j < KERNY; j++){ 
					#pragma omp unroll
					for(int i = 0; i < KERNX; i++){ 					
						tmp += zeropaddedKernel[i+j*newKERNX] * zeropadded[(x+i) + (y+j)*newX];														
					}									
				}
				out[x+y*data_size_X] = tmp;
			}
		}
	}
	#pragma omp end parallel

	free(copy_out);
	return 1;

}
