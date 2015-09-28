#include <emmintrin.h>
#define KERNX 3 //this is the x-size of the kernel. It will always be odd.
#define KERNY 3 //this is the y-size of the kernel. It will always be odd.

int conv2D(float* in, float* out, int data_size_X, int data_size_Y,
                    float* kernel)
{ 
    int kern_cent_X = (KERNX - 1)/2;
    int kern_cent_Y = (KERNY - 1)/2;
    int newKERNX = KERNX+(4-KERNX%4);   
    float zeropaddedKernel[newKERNX*KERNY]; 

	for(int z = 0; z < newKERNX*KERNY; z++)     
		zeropaddedKernel[z] = 0;

    for(int q = -kern_cent_Y; q <= kern_cent_Y; q++) {
		for(int w = -kern_cent_X; w <= kern_cent_X; w++) {
			zeropaddedKernel[(w+kern_cent_X)+(q+kern_cent_Y)*newKERNX] = kernel[(kern_cent_X-w)+(kern_cent_Y-q)*KERNX];   
    	}
    }


    int newX = data_size_X+((KERNX/2)*2); 
    int newY = data_size_Y+((KERNX/2)*2); 
	newX += (4-(newX%4));
	
	float zeropadded[newX*newY]; 
	for(int z = 0; z < newX*newY; z++)     
		zeropadded[z] = 0;       

    for(int b = 0; b < data_size_Y; b++){
		for(int a = 0; a < data_size_X; a++){
			zeropadded[(a+(KERNX/2))+(b+(KERNX/2))*newX] = in[a + b*data_size_X];
		}
	}
	
   	   	
	//C[j*M + i] += A[k*M + i] * B[j*M + k];  jki   	   	
	float tmp, tmp2;
	for(int y = 0; y < data_size_Y ; y++){ 
		for(int x = 0; x < data_size_X ; x++){ 
			tmp = 0.0;
			for(int j = 0; j < KERNY; j++){ 
				for(int i = 0; i < KERNX; i++){ 					
					tmp2 = zeropaddedKernel[i+j*newKERNX];													
					tmp += tmp2 * zeropadded[(x+i) + (y+j)*newX];								
					//out[x+y*data_size_X] += kernel[(kern_cent_X-i)+(kern_cent_Y-j)*KERNX] * in[(x+i) + (y+j)*data_size_X];
				} 
				out[x+y*data_size_X] = tmp;
			}
		}
	}

	return 1;
}

