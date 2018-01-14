/*
 ( ͡° ͜ʖ ͡°) °º¤ø,¸¸,ø¤º°`°º¤ø,¸,ø¤°º¤ø,¸¸,ø¤º°`°º¤ø,¸,ø¤º°`
 
 Currently this file is under the BSD 3-clause license.
 If you use this code, include the license (yes including the lenny faces).
 All of my licenses are of course negotiable, so contact me if you actually need this.
 
 ( ͡° ͜ʖ ͡°) °º¤ø,¸¸,ø¤º°`°º¤ø,¸,ø¤°º¤ø,¸¸,ø¤º°`°º¤ø,¸,ø¤º°`
 
 Copyright (c) 2018 (9 jan), Kevin De Keyser
 All rights reserved.
 
 ( ͡° ͜ʖ ͡°) °º¤ø,¸¸,ø¤º°`°º¤ø,¸,ø¤°º¤ø,¸¸,ø¤º°`°º¤ø,¸,ø¤º°`
 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:
 * Redistributions of source code must retain the above copyright
 notice, this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright
 notice, this list of conditions and the following disclaimer in the
 documentation and/or other materials provided with the distribution.
 * The name of its contributors may not be used to endorse or promote products
 derived from this software without specific prior written permission.
 
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

//For best performance use a compiler that can vectorize the low fft no. count.
//Can do any radix FFT (using chirp-z transformation), but might be slow for non-radix 2.

//Missing but possibly good optimisations:
//- Mix fft3 with fft2 properly. Currently only does fft3 for input of size 1536.
//- Use real FFT if you plan to do convolution or another operation, where you know that the input values are real numbers.
//- Vectorization
//- Optimise the small cases (n=2,3,4,6,8,12).
//- Iterative? IMO this makes the code just less readable and prob gives an increase of 5% or so.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

typedef struct {
    float re;
    float im;
} FComplex; //floating point complex

static const float TWO_PI = 6.28318530717958647692f; //will be rounded accordingly

static void init_twiddles(int size);
static void fft2(FComplex * out, FComplex *in, int size, int fact);
static void ifft(FComplex * out, FComplex *in, int size);
static void chirpzfft(FComplex * out, FComplex *in, int size);

//These are the external functions, which are supposed to be called by the tester!
extern void timefft(FComplex * out, FComplex * in, int size);
extern void timeifft(FComplex * out, FComplex * in, int size);

static inline FComplex fcadd(FComplex a, FComplex b);
static inline FComplex fcsub(FComplex a, FComplex b);
static inline FComplex fcmul(FComplex a, FComplex b);
static inline FComplex fcmulgauss(FComplex a, FComplex b);
static inline FComplex fcreciproc(FComplex a);




// DEBUG METHODS
#include <stdio.h>

static void printfcomplex(FComplex a) {
    printf("%lf %lfi\n", a.re, a.im);
}

// END OF DEBUG METHODS


static inline FComplex fcadd(FComplex a, FComplex b) {
    FComplex c;
    c.re = a.re + b.re;
    c.im = a.im + b.im;
    return c;
}
static inline FComplex fcsub(FComplex a, FComplex b) {
    FComplex c;
    c.re = a.re - b.re;
    c.im = a.im - b.im;
    return c;
}
static inline FComplex fcmul(FComplex a, FComplex b) {
    //(a+bi)*(c+di) = (a*c-b*d)+(a*d+b*c)i
    FComplex c;
    c.re = a.re * b.re - a.im * b.im;
    c.im = a.re * b.im + a.im * b.re;
    return c;
}
//Using gauss multiplication a single multiplication can be ommitted, however the performance increase is hardware dependent.
static inline FComplex fcmulgauss(FComplex a, FComplex b) {
    FComplex c;
    double t1 = a.re * b.re;
    double t2 = a.im * b.im;
    double t3 = (a.re+a.im) * (b.re+b.im);
    c.re = t1-t2;
    c.im = t3-t1-t2;
    return c;
}

static inline FComplex fcreciproc(FComplex a) {
    FComplex ainv;
    float normsq = a.re*a.re + a.im * a.im;
    ainv.re = a.re / normsq;
    ainv.im = -a.im / normsq;
    return ainv;
}



//twiddle factors
static float * sintable = NULL; //sintable[i] = sin(TWO_PI*i/size)
static float * costable = NULL; //costable[i] = cos(TWO_PI*i/size)
static float * cosminsintable = NULL; //cosminsintable[i] = cos(TWO_PI*i/size) - sin(TWO_PI*i/size)
static float * cosplussintable = NULL; //cosplussintable[i] = cos(TWO_PI*i/size) + sin(TWO_PI*i/size)
static int prevsize = 0;

static void init_twiddles(int size) {  // Initialize FFT twiddle factors
    if(size == prevsize) return;
    
    free(sintable);
    free(costable);
    free(cosminsintable);
    free(cosplussintable);
    
    //Allocate both sine and cosine table
    if(!(sintable = malloc(sizeof(float) * size))) {fprintf(stderr, "Out of memory!\n"); exit(1);}
    if(!(costable = malloc(sizeof(float) * size))) {fprintf(stderr, "Out of memory!\n"); exit(1);}
    if(!(cosminsintable = malloc(sizeof(float) * size))) {fprintf(stderr, "Out of memory!\n"); exit(1);}
    if(!(cosplussintable = malloc(sizeof(float) * size))) {fprintf(stderr, "Out of memory!\n"); exit(1);}
    
    for(int i = 0; i < size; ++i) {
        sintable[i] = sin(TWO_PI*i/size);
        costable[i] = cos(TWO_PI*i/size);
        
        cosminsintable[i] = costable[i] - sintable[i];
        cosplussintable[i] = costable[i] + sintable[i];
    }
}



//O(n^2) implementation for fft, sometimes is quite fast.
static void slowfft(FComplex *out, FComplex *in, int size) {
    for(int k = 0; k < size; ++k) {
        out[k].im = 0; out[k].re = 0;
        for(int j = 0; j < size; ++j) {
            FComplex w = {cos(TWO_PI/size * k * j), -sin(TWO_PI/size * k * j)};
            out[k] = fcadd(out[k],fcmul(in[j],w));
        }
    }
}

static void slowifft(FComplex *out, FComplex *in, int size) {
    for(int k = 0; k < size; ++k) {
        out[k].im = 0; out[k].re = 0;
        for(int j = 0; j < size; ++j) {
            FComplex w = {cos(TWO_PI/size * k * j), sin(TWO_PI/size * k * j)};
            out[k] = fcadd(out[k],fcmul(in[j],w));
        }
        out[k].re = 1./size * out[k].re;
        out[k].im = 1./size * out[k].im;
    }
}



//fft2 only works for power of 2 arrays. Use chirpzfft instead for other sizes.
//Make sure to call init_twiddles(size) before calling fft2 (see timefft2 for usage).
static void fft2(FComplex * out, FComplex *in, int size, int fact) {
    if(size==1) {out[0] = in[0]; return;}
    //assumes that size is power of 2
    int halfsize = size/2, doublefact = fact*2;
    fft2(out, in, halfsize, doublefact); //even
    fft2(out+fact, in+fact, halfsize, doublefact); //odd
    FComplex outres [size];
    for(int k = 0; k < halfsize; ++k) {
        //We need to compute:
        //y[k] = yeven[k] + yodd[k] * w_size^k
        //y[k+halfsize] = yeven[k] - yodd[k] * w_size^k
        //However since out contains both the result of yeven/yodd and should later contain the result of the fft, an overlap happens!
        //Therefore the results are stored in the outres vector and later applied to out.
        
        //1. Compute yodd[k] * w^k
        //Idea 1: You can precompute w^k = costable[fact*k] - sintable[fact*k] * i
        //Idea 2: Instead of naively multiplying w^k * yodd[k] we can do the following:
        //Normal multiplication equation:
        //(a+bi)*(cos+sin*i) = (a*cos-b*sin)+(b*cos+a*sin)*i = X+Yi
        //Equation using only 3 multiplies:
        //I=cos*(a−b)
        //J=cos+sin
        //K=cos−sin
        //X=J*b+I
        //Y=K*a−I
        //Now realise that J and K can be precomputed (cosminsintable and cosplussintable)!
        
        //Fast method:
        FComplex yodd = out[fact+doublefact*k];
        FComplex t;
        int wind = fact*k;
        double I = costable[wind]*(yodd.re - yodd.im);
        t.re = cosplussintable[wind] * yodd.im + I;
        t.im = cosminsintable[wind]  * yodd.re - I;
        
        outres[k] = fcadd(out[doublefact*k],t);
        outres[k+halfsize] = fcsub(out[doublefact*k],t);
    }
    for(int k = 0; k < halfsize; ++k) {
        out[fact*k] = outres[k];
        out[fact*(k+halfsize)] = outres[k+halfsize];
    }
}

//This method computes czt(in), see: https://mathworks.com/help/signal/ref/czt.html
//czt(x) can compute fft of non base-2 numbers, by using a radix2 fft as subalgorithm.
static void chirpzfft(FComplex * out, FComplex *in, int size) {
    int npow2 = 2 * size - 1;
    //Compute next 2-power of 2*size-1:
    npow2--;
    npow2 |= npow2 >> 1;
    npow2 |= npow2 >> 2;
    npow2 |= npow2 >> 4;
    npow2 |= npow2 >> 8;
    npow2 |= npow2 >> 16;
    npow2++;
    
    FComplex chirp [2*size-1];
    for(int i = 1 - (int)size; i < size; ++i) {
        float exponent = i*i/2.;
        FComplex val;
        val.re = cos(TWO_PI/size * exponent);
        val.im = -sin(TWO_PI/size * exponent);
        chirp[i-(1-size)] = val;
    }
    
    FComplex xp [npow2];
    for(int i = 0; i < size; ++i) xp[i] = fcmul(in[i], chirp[size-1+i]);
    for(int i = size; i < npow2; ++i) {xp[i].re = 0; xp[i].im = 0;} //make sure xp is zero padded after size
    
    FComplex recipchirp[npow2]; //holds reciprocal of chirpz
    for(int i = 0; i < 2*size-1; ++i) recipchirp[i] = fcreciproc(chirp[i]);
    for(int i = 2*size; i < npow2; ++i) {recipchirp[i].re = 0; recipchirp[i].im = 0;} //make sure the rest is zero padded
    
    FComplex arr1 [npow2];
    FComplex arr2 [npow2];
    
    init_twiddles(npow2);
    
    fft2(arr1, xp, npow2, 1);
    fft2(arr2, recipchirp, npow2, 1);
    
    for(int i = 0; i < npow2; ++i) arr1[i] = fcmulgauss(arr1[i],arr2[i]); //make arr1 the element-wise product.
    ifft(arr2,arr1,npow2); //store ifft(fft(xp)*fft(recichirp)) into arr2
    
    for(int i = 0; i < size; ++i) out[i] = fcmulgauss(chirp[size-1+i], arr2[size-1+i]);
}

static void fft1536(FComplex * out, FComplex * in) {
    //specially crafted for FFT-1536
    FComplex outres [1536];
    
    init_twiddles(1536);
    fft2(outres, in  , 512, 3);
    fft2(outres+1, in+1, 512, 3);
    fft2(outres+2, in+2, 512, 3);
    
    for(int i = 0; i < 1536; ++i) {
        FComplex w1;
        w1.re = costable[i]; //cos(TWO_PI * i / 1536.);
        w1.im = -sintable[i];
        FComplex w2;
        if(i*2 < 1536) {
            w2.re = costable[i*2];
            w2.im = -sintable[i*2];
        } else {
            w2.re = cos(TWO_PI * i / 768.);
            w2.im = -sin(TWO_PI * i / 768.);
        }
        out[i] = fcadd(outres[3*(i%512)], fcmulgauss(outres[3*(i%512)+1], w1));
        out[i] = fcadd(out[i], fcmulgauss(outres[3*(i%512)+2], w2));
    }
}

//This methods computes the ifft of in and stores it into out. If size is a power of 2 it will directly call fft2, elsewise it will compute the fft using chirpzfft.
static void ifft(FComplex *out, FComplex *in, int size) {  // Perform out = FFT(in)
    FComplex in2 [size];
    
    //store the conjugate in in2
    for(int i = 0; i < size; ++i) {
        in2[i].re = in[i].re;
        in2[i].im = -in[i].im;
    }
    
    if((size & (size - 1)) == 0) { //if power of 2
        init_twiddles(size);
        fft2(out, in2, size, 1);
    }
    else if(size == 1536) {
        fft1536(out, in2);
    }
    else { //if not power of 2
        chirpzfft(out, in2, size);
    }
    
    //back conjugate the result and multiply it with 1./size
    for(int i = 0; i < size; ++i) {
        out[i].re = 1./size * out[i].re;
        out[i].im = - 1./size * out[i].im;
    }
}

// Timing of fft/ifft can be done using the methods timefft/timeifft. Currently this uses clock_t. Alternatively use the following instruction for x86:

static unsigned long long rdtscl(void) {
    unsigned int lo, hi;
    __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
    return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}


extern void timefft(FComplex *out, FComplex *in, int size) {  // Perform out = FFT(in)
    if(size <= 0) return;
    if((size & (size - 1)) == 0) { //if power of 2, no chirp z transformation necessairy and can directly use radix2.
        unsigned long long startcycles, midcycles, endcycles;
        clock_t starttime, midtime, endtime;
        starttime = clock();
        startcycles = rdtscl();
        init_twiddles(size);
        midtime = clock();
        midcycles = rdtscl();
        fft2(out,in,size,1);
        endcycles = rdtscl();
        endtime = clock();
        printf("Computer cycles (measured using rdtscl instr.) with twiddling: %lld, without twiddling: %lld.\n", endcycles-startcycles, midcycles-startcycles);
        printf("Clock ticks used for radix2 fft with twiddling: %ld, without twiddling: %ld.\n", endtime-starttime, endtime-midtime);
        printf("Seconds used for radix2 fft with twiddling: %lf(s), without twiddling: %lf(s).\n", (double)(endtime-starttime)/CLOCKS_PER_SEC, (double)(midtime-starttime)/CLOCKS_PER_SEC);
    } else if(size == 1536) { //does a single radix-3 pass and then does radix-2.
        unsigned long long startcycles, endcycles;
        clock_t starttime, endtime;
        starttime = clock();
        startcycles = rdtscl();
        fft1536(out, in);
        endcycles = rdtscl();
        endtime = clock();
        printf("Computer cycles (measured using rdtscl instr.) for 1536 sized fft: %lld.\n", endcycles-startcycles);
        printf("Clock ticks used for 1536 sized fft: %ld.\n", endtime-starttime);
        printf("Seconds used for 1536 sized fft: %lf(s).\n", (double)(endtime-starttime)/CLOCKS_PER_SEC);
    } else if(size < 64) { //for small sizes, which are not a power of 2, it is faster to use O(n^2) solution.
        unsigned long long startcycles, endcycles;
        clock_t starttime, endtime;
        starttime = clock();
        startcycles = rdtscl();
        slowfft(out, in, size);
        endcycles = rdtscl();
        endtime = clock();
        printf("Computer cycles (measured using rdtscl instr.) for slow-fft: %lld.\n", endcycles-startcycles);
        printf("Clock ticks used for slow-fft: %ld.\n", endtime-starttime);
        printf("Seconds used for slow-fft: %lf(s).\n", (double)(endtime-starttime)/CLOCKS_PER_SEC);
    } else {
        unsigned long long startcycles, endcycles;
        clock_t starttime, endtime;
        starttime = clock();
        startcycles = rdtscl();
        chirpzfft(out, in, size);
        endcycles = rdtscl();
        endtime = clock();
        printf("Computer cycles (measured using rdtscl instr.) for chirpz-fft: %lld.\n", endcycles-startcycles);
        printf("Clock ticks used for chirpz-fft: %ld.\n", endtime-starttime);
        printf("Seconds used for chirpz-fft: %lf(s).\n", (double)(endtime-starttime)/CLOCKS_PER_SEC);
    }
}


extern void timeifft(FComplex *out, FComplex *in, int size) {  // Perform out = FFT(in)
    if(size <= 0) return;
    if((size & (size - 1)) == 0) { //if power of 2 no chirp z transformation necessairy
        unsigned long long startcycles, endcycles;
        clock_t starttime, endtime;
        starttime = clock();
        startcycles = rdtscl();
        ifft(out,in,size); //will use radix2-fft as subroutine.
        endcycles = rdtscl();
        endtime = clock();
        printf("Computer cycles (measured using rdtscl instr.) for radix2-ifft: %lld.\n", endcycles-startcycles);
        printf("Clock ticks used for radix2-ifft: %ld.\n", endtime-starttime);
        printf("Seconds used for radix2-ifft: %lf(s).\n", (double)(endtime-starttime)/CLOCKS_PER_SEC);
    } else if(size == 1536) {  //does a single 3-radix pass and then does 2-radix
        unsigned long long startcycles, endcycles;
        clock_t starttime, endtime;
        starttime = clock();
        startcycles = rdtscl();
        ifft(out,in,size); //will use 1536-fft as subroutine.
        endcycles = rdtscl();
        endtime = clock();
        printf("Computer cycles (measured using rdtscl instr.) for 1536 sized ifft: %lld.\n", endcycles-startcycles);
        printf("Clock ticks used for 1536 sized ifft: %ld.\n", endtime-starttime);
        printf("Seconds used for 1536 sized ifft: %lf(s).\n", (double)(endtime-starttime)/CLOCKS_PER_SEC);
    } else if(size < 64) { //for small sizes, which are not a power of 2, it is faster to use O(n^2) solution.
        unsigned long long startcycles, endcycles;
        clock_t starttime, endtime;
        starttime = clock();
        startcycles = rdtscl();
        slowifft(out, in, size);
        endcycles = rdtscl();
        endtime = clock();
        printf("Computer cycles (measured using rdtscl instr.) for slow-ifft: %lld.\n", endcycles-startcycles);
        printf("Clock ticks used for slow-ifft: %ld.\n", endtime-starttime);
        printf("Seconds used for slow-ifft: %lf(s).\n", (double)(endtime-starttime)/CLOCKS_PER_SEC);
    } else {
        unsigned long long startcycles, endcycles;
        clock_t starttime, endtime;
        starttime = clock();
        startcycles = rdtscl();
        ifft(out, in, size); //will use chirpz-fft as subroutine.
        endcycles = rdtscl();
        endtime = clock();
        printf("Computer cycles (measured using rdtscl instr.) for chirpz-ifft: %lld.\n", endcycles-startcycles);
        printf("Clock ticks used for chirpz-ifft: %ld.\n", endtime-starttime);
        printf("Seconds used for chirpz-ifft: %lf(s).\n", (double)(endtime-starttime)/CLOCKS_PER_SEC);
    }
}


int main(int argc, const char * argv[]) {
    printf("Please run multiple times to measure # of cycles, since they can easily vary up to a factor of 2 on my machine!\n");
    //This is the FComplex -> FComplex way of using it (most recommended):
    printf("Example of length 1536 using FComplex -> FComplex (most recommended)\n");
    
    FComplex * arrin = calloc(1536, sizeof(FComplex));
    FComplex * arrres = calloc(1536, sizeof(FComplex));
    FComplex * arrres2 = calloc(1536, sizeof(FComplex));
    
    if(!arrin || !arrres || !arrres2) { fprintf(stderr, "Error: Cannot allocate memory."); exit(1); }
    
    arrin[0].re = 0.45; arrin[0].im = 0;
    arrin[1].re = 0.45; arrin[1].im = -0.1;
    arrin[2].re = 0.45; arrin[2].im = +0.1;
    arrin[3].re = 0.45; arrin[3].im = -0.2;
    arrin[4].re = 0.45; arrin[4].im = +0.2;
    arrin[5].re = 0.45; arrin[5].im = -0.3;
    arrin[6].re = 0.45; arrin[6].im = +0.3;
    arrin[7].re = 0.45; arrin[7].im = -0.4;
    arrin[8].re = 0.45; arrin[8].im = +0.4;
    arrin[9].re = 0.45; arrin[10].im = -0.5;
    arrin[10].re = 0.45; arrin[11].im = +0.5;
    arrin[11].re = 0.45; arrin[12].im = -0.6;
    arrin[12].re = 0.45; arrin[13].im = +0.6;
    arrin[13].re = 0.45; arrin[14].im = -0.7;
    arrin[14].re = 0.45; arrin[15].im = +0.7;
    arrin[15].re = 0.55; arrin[15].im = -0.15;
    arrin[16].re = 0.25; arrin[16].im = -0.25;
    
    
    for(int i = 17; i < 1536; ++i) {
        arrin[17].re = 1./(i%30);
        arrin[17].im = -1./(i%30);
    }
    printf("------------\n");
    timefft(arrres,arrin, 1536);
    printf("------------\n");
    timeifft(arrres2,arrin, 1536);
    printf("------------\n");
    
    printf("First 50 entries of FFT:\n");
    for(int i = 0; i < 50; ++i) printfcomplex(arrres[i]);
    
    
    return 0;
}


