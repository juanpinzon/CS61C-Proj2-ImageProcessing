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
	float tmp;  

	int rowMin, rowMax;                             // to check boundary of input array
    int colMin, colMax;
    
    //kern_cent_X = kCenterX 
    //kern_cent_Y = kCenterY
    float *inPtr, *inPtr2, *outPtr, *kPtr;
    // init working  pointers
    inPtr = inPtr2 = &zeropadded[data_size_X * kern_cent_Y + kern_cent_X];  // note that  it is shifted (kCenterX, kCenterY),
    outPtr = out;
    kPtr = zeropaddedKernel;
    int i, j, m, n;
	
	for(i = 0; i < data_size_Y ; i++){ 
	 	rowMax = i + kern_cent_Y;
        rowMin = i - data_size_Y + kern_cent_Y;
        
		for(j = 0; j < data_size_X ; j++){ 
			//tmp = 0.0;
			colMax = j + kern_cent_X;
            colMin = j - data_size_X + kern_cent_X;

            *outPtr = 0;                            // set to 0 before accumulate
            
			// flip the kernel and traverse all the kernel values
            // multiply each kernel value with underlying input data
            for(m = 0; m < KERNY; ++m)        // kernel rows
            {
                // check if the index is out of bound of input array
                if(m <= rowMax && m > rowMin)
                {
                    for(n = 0; n < KERNX; ++n)
                    {
                        // check the boundary of array
                        if(n <= colMax && n > colMin)
                            *outPtr += *(inPtr - n) * *kPtr;
                        ++kPtr;                     // next kernel
                    }
                }
                else
                    kPtr += KERNX;            // out of bound, move to next row of kernel

                inPtr -= data_size_X;                 // move input data 1 raw up
            }

            kPtr = zeropaddedKernel;                          // reset kernel to (0,0)
            inPtr = ++inPtr2;                       // next input
            ++outPtr;                               // next output
			
		}
	}

	return 1;
}

