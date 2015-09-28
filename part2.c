#include <emmintrin.h> 
#include <omp.h>
#define KERNX 3 //this is the x-size of the kernel. It will always be odd.
#define KERNY 3 //this is the y-size of the kernel. It will always be odd.
#define NUM_THREADS 8
int conv2D(float* in, float* out, int data_size_X, int data_size_Y,
                    float* kernel)

{ 
	int kern_cent_X = (KERNX - 1)/2;
	int kern_cent_Y = (KERNY - 1)/2;

	omp_set_num_threads(NUM_THREADS);  
	float zeropaddedKernel[KERNX*KERNY];
	for(int z = 0; z < KERNX*KERNY; z++)    
		zeropaddedKernel[z] = 0;

	for(int q = -kern_cent_Y; q <= kern_cent_Y; q++) {
		for(int w = -kern_cent_X; w <= kern_cent_X; w++) {
			zeropaddedKernel[(w+kern_cent_X)+(q+kern_cent_Y)*KERNX] = kernel[(kern_cent_X-w)+(kern_cent_Y-q)*KERNX];   
		}
	}

	int newX = data_size_X+((KERNX/2)*2); 
	int newY = data_size_Y+((KERNX/2)*2); 
	newX += (4-(newX%4));
	float zeropadded[newX*newY]; 

	#pragma omp parallel for firstprivate(newX, newY)
	for(int z = 0; z < newX*newY; z++)     
		zeropadded[z] = 0;       

	int a,b;
	#pragma omp parallel for firstprivate(newX,data_size_X, data_size_Y)
	for(b = 0; b < data_size_Y; b++){
		for(a = 0; a < data_size_X; a++){
			zeropadded[(a+(KERNX/2))+(b+(KERNX/2))*newX] = in[a + b*data_size_X];
		}
	}

	#pragma omp parallel 
	{
		__m128 Kernel1;
		__m128 Kernel2;
		__m128 Kernel3;

		__m128 input1 = _mm_setzero_ps();
		__m128 input2 = _mm_setzero_ps();
		__m128 input3 = _mm_setzero_ps();

		__m128 tmp = _mm_setzero_ps();

		int y,x,j,i;
 		#pragma omp for schedule(dynamic, NUM_THREADS) nowait 
		//#pragma omp parallel for firstprivate(tmp, newX, data_size_X, data_size_Y)
		for(y = 0; y < data_size_Y ; y++){
			for(x = 0; x < data_size_X/4*4 ; x+=4){ 
				tmp = _mm_setzero_ps();
				for(j = 0; j < KERNY; j++){ 
					for(i = 0; i < KERNX; i+=3){
						input1 = _mm_loadu_ps(zeropadded + (x+i)+(y+j)*newX);
						input2 = _mm_loadu_ps(zeropadded + (x+i+1)+(y+j)*newX);
						input3 = _mm_loadu_ps(zeropadded + (x+i+2)+(y+j)*newX);

						Kernel1 = _mm_load1_ps(zeropaddedKernel + i+j*KERNX);	
						Kernel2 = _mm_load1_ps(zeropaddedKernel + i+1+j*KERNX);	
						Kernel3 = _mm_load1_ps(zeropaddedKernel + i+2+j*KERNX);	
						tmp = _mm_add_ps(tmp, _mm_mul_ps(input1, Kernel1));
						tmp = _mm_add_ps(tmp, _mm_mul_ps(input2, Kernel2));
						tmp = _mm_add_ps(tmp, _mm_mul_ps(input3, Kernel3));
					}
				}
				_mm_storeu_ps(out + x+y*data_size_X, tmp);
			}
		}

		#pragma omp for schedule(dynamic, NUM_THREADS) nowait 
		for(y = 0; y < data_size_Y ; y++){
			for(x = data_size_X/4*4; x < data_size_X ; x++){ 
				for(j = 0; j < KERNY; j++){ 
					for(i = 0; i < KERNX; i++){
						out[x+y*data_size_X] += zeropaddedKernel[i+j*KERNX] * zeropadded[(x+i) + (y+j)*newX];
					}
				}
			}
		}
	} 

return 1;
}
