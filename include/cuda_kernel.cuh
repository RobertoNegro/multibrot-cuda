#ifndef MULTIBROT_CUDA_KERNEL_CUH
#define MULTIBROT_CUDA_KERNEL_CUH

void multibrot(
        unsigned int unroll,
        unsigned int blockSize,
        unsigned char *rgb,
        int width, int height,
        int exponent, int iterations, double R, double eps,
        unsigned char borderR, unsigned char borderG, unsigned char borderB, double borderThickness,
        long normOrbitSkip, double normLightIntensity, double normLightAngle, double normLightHeight,
        unsigned char bgR, unsigned char bgG, unsigned char bgB,
        double kR, double kG, double kB, double kD,
        unsigned char internalBorderR, unsigned char internalBorderG, unsigned char internalBorderB,
        unsigned char internalCoreR, unsigned char internalCoreG, unsigned char internalCoreB, double internalK,
        double stripeDensity, double stripeLightIntensity,
        double zoom, double posX, double posY);

#endif