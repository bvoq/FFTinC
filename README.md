# FFTinC
This is a FFT implementation in C that works with any input size and uses the Chirp-Z transformation for input sizes, which are not radix 2.

**Sample usage:**

    FComplex * x = calloc(5, sizeof(FComplex));
    FComplex * yfwd = calloc(5, sizeof(FComplex));
    FComplex * yinv = calloc(5, sizeof(FComplex));
    if(!x || !yfwd || !yinv) { fprintf(stderr, "Error: Cannot allocate memory."); exit(1); }

    x[0].re = 7.65; x[0].im = 0;
    x[1].re = 0.45; x[1].im = -0.1;
    x[2].re = 35; x[2].im = +0.1;
    x[3].re = 0.15; x[3].im = -2;
    x[4].re = 2.4; x[4].im = +0.2;
    
    printf("------------\n");
    timefft(yfwd,arrin, 1536);
    printf("------------\n");
    timeifft(yinv,arrin, 1536);
     printf("------------\n");
    
    printf("Computed size 5-FFT:\n");
    for(int i = 0; i < 5; ++i) printfcomplex(yfwd[i]);
    
    printf("Computed size 5-IFFT:\n");
    for(int i = 0; i < 5; ++i) printfcomplex(yinv[i]);
