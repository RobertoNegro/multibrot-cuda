#define _USE_MATH_DEFINES

#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <cmath>
#include <cuda_profiler_api.h>

using namespace std;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__
void multibrot_kernel(
        unsigned int unroll,
        unsigned char *image,
        int width, int height, double ratio,
        int exponent, int iterations, double R, double eps,
        unsigned char borderR, unsigned char borderG, unsigned char borderB, double borderThickness,
        long normOrbitSkip, double normLightIntensity, double normLightAngle, double normLightHeight,
        unsigned char bgR, unsigned char bgG, unsigned char bgB,
        double kR, double kG, double kB, double kD,
        unsigned char internalBorderR, unsigned char internalBorderG, unsigned char internalBorderB,
        unsigned char internalCoreR, unsigned char internalCoreG, unsigned char internalCoreB,
        double internalK,
        double stripeDensity, double stripeLightIntensity,
        double zoom, double posX, double posY
) {
    unsigned int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    for (unsigned int unrollIndex = 0; unrollIndex < unroll; unrollIndex++) {
        unsigned int currentIndex = threadIndex * unroll + unrollIndex;
        if (currentIndex >= width * height) {
            return;
        }

        //region Calculations
        double c_r = (((currentIndex % width - 1) - width / 2.) / (width * zoom)) * ratio + posX;
        double c_i = ((double) currentIndex / width - height / 2.) / (height * zoom) + posY;

        double z_r = c_r;
        double z_i = c_i;

        double last_z_r = 0;
        double last_z_i = 0;

        double dz_r = 1.;
        double dz_i = 0.;

        double dc_r = 1.;
        double dc_i = 0.;

        double dzdz_r = 0.;
        double dzdz_i = 0.;

        double dcdc_r = 0.;
        double dcdc_i = 0.;

        double dcdz_r = 0.;
        double dcdz_i = 0.;

        double p = 1.;

        double orbitCount = 0;

        double V = 0;

        long i;
        for (i = 0; i < iterations; i++) {
            double z2 = z_r * z_r + z_i * z_i;
            if (z2 > R * R) {
                V = log(z2) / p;
                break;
            }

            if (eps > 0 && dz_r * dz_r + dz_i * dz_i < eps * eps) {
                V = 0;
                break;
            }

            double dzdz_r_temp = 2 * ((z_r * dzdz_r - z_i * dzdz_i) + (dz_r * dz_r - dz_i * dz_i));
            dzdz_i = 2 * ((z_r * dzdz_i + z_i * dzdz_r) + (dz_r * dz_i + dz_i * dz_r));
            dzdz_r = dzdz_r_temp;

            double dcdc_r_temp = 2 * ((z_r * dcdc_r - z_i * dcdc_i) + (dc_r * dc_r - dc_i * dc_i));
            dcdc_i = 2 * ((z_r * dcdc_i + z_i * dcdc_r) + (dc_r * dc_i + dc_i * dc_r));
            dcdc_r = dcdc_r_temp;

            double dcdz_r_temp = 2 * ((z_r * dcdz_r - z_i * dcdz_i) + (dz_r * dc_r - dz_i * dc_i));
            dcdz_i = 2 * ((z_r * dcdz_i + z_i * dcdz_r) + (dc_r * dz_i + dc_i * dz_r));
            dcdz_r = dcdz_r_temp;

            double dz_r_temp = 2 * (z_r * dz_r - z_i * dz_i);
            dz_i = 2 * (z_r * dz_i + z_i * dz_r);
            dz_r = dz_r_temp;

            double dc_r_temp = 2 * (z_r * dc_r - z_i * dc_i) + 1;
            dc_i = 2 * (z_r * dc_i + z_i * dc_r);
            dc_r = dc_r_temp;

            p *= 2.;

            if (i >= normOrbitSkip) {
                orbitCount += 0.5 + 0.5 * sin(stripeDensity * atan2(last_z_i, last_z_r));
            }
            last_z_r = z_r;
            last_z_i = z_i;

            int esp = exponent;
            if (esp != 0) {
                if (esp < 0) {
                    esp = -esp;

                    double z_r_temp = z_r / (z_r * z_r + z_i * z_i);
                    z_i = -z_i / (z_r * z_r + z_i * z_i);
                    z_r = z_r_temp;
                }
                double z_esp_r = z_r;
                double z_esp_i = z_i;
                for (int e = 1; e < esp; e++) {
                    double z_esp_r_temp = (z_r * z_esp_r - z_i * z_esp_i);
                    z_esp_i = (z_esp_i * z_r + z_i * z_esp_r);
                    z_esp_r = z_esp_r_temp;
                }
                z_r = z_esp_r + c_r;
                z_i = z_esp_i + c_i;
            } else {
                z_r = 1.0;
                z_i = 0.0;
            }

        }
        // endregion

        if (V == 0) { // Inside!
            //region Interior distance estimation
            double u_r = (dzdz_r * dc_r - dzdz_i * dc_i);
            double u_i = (dzdz_r * dc_i + dzdz_i * dc_r);
            double v_r = 1 - dz_r;
            double v_i = -dz_i;

            double u_r_temp = (u_r * v_r + u_i * v_i) / (v_r * v_r + v_i * v_i);
            u_i = (u_i * v_r - u_r * v_i) / (v_r * v_r + v_i * v_i);
            u_r = u_r_temp;

            u_r = u_r + dcdz_r;
            u_i = u_i + dcdz_i;

            double d = (1. - (dz_r * dz_r + dz_i * dz_i)) / sqrt(u_r * u_r + u_i * u_i);
            //endregion

//        if (d < 1) {
//            image[currentIndex * 4] = 0;
//            image[currentIndex * 4 + 1] = (int) max(0., min(255., (255. * tanh(d))));
//            image[currentIndex * 4 + 1] = (unsigned char) (max(0., min(255., 0 + d * (255 - 0))));
//            image[currentIndex * 4 + 2] = 0;
//        } else {
//            image[currentIndex * 4] = 0;
//            image[currentIndex * 4 + 1] = 255;
//            image[currentIndex * 4 + 2] = 0;
//        }

            double mix = internalK > 0 ? log(d) / internalK : 1;
            if (mix < 1) {
                image[currentIndex * 4] = max(0., min(255., internalBorderR + mix * (internalCoreR - internalBorderR)));
                image[currentIndex * 4 + 1] = max(0., min(255., internalBorderG + mix * (internalCoreG - internalBorderG)));
                image[currentIndex * 4 + 2] = max(0., min(255., internalBorderB + mix * (internalCoreB - internalBorderB)));
            } else {
                image[currentIndex * 4] = internalCoreR;
                image[currentIndex * 4 + 1] = internalCoreG;
                image[currentIndex * 4 + 2] = internalCoreB;
            }
        } else { // Outside!
            //region Exterior distance estimation
            double rad = sqrt(z_r * z_r + z_i * z_i);
            double d = rad * 2. * log(rad) / sqrt(dc_r * dc_r + dc_i * dc_i);
            //endregion

            unsigned char tempR = bgR;
            unsigned char tempG = bgG;
            unsigned char tempB = bgB;

            //region Gradient Background Setup
            if (kR > 0) {
                tempR = (unsigned char) (max(0., min(255., tempR + (255. * (1 + cos(M_PI_2 * log(V) / (kR))) / 2. / kD))));
            }
            if (kG > 0) {
                tempG = (unsigned char) (max(0., min(255., tempG + (255. * (1 + cos(M_PI_2 * log(V) / (kG))) / 2. / kD))));
            }
            if (kB > 0) {
                tempB = (unsigned char) (max(0., min(255., tempB + (255. * (1 + cos(M_PI_2 * log(V) / (kB))) / 2. / kD))));
            }
            //endregion

            //region 3D Normal
            if (normLightIntensity != 1) {
                double vR = cos(normLightAngle * 2. * M_PI / 360.);
                double vI = sin(normLightAngle * 2. * M_PI / 360.);
                double lo = 0.5 * log(z_r * z_r + z_i * z_i);
                double conjR = ((1. + lo) * (dc_r * dc_r - dc_i * dc_i) - (lo) * (z_r * dcdc_r - z_i * dcdc_i));
                double conjI = ((1. + lo) * -(dc_r * dc_i + dc_i * dc_r) - (lo) * -(z_r * dcdc_i + z_i * dcdc_r));
                double uR = (z_r * dc_r - z_i * dc_i);
                double uI = (z_r * dc_i + z_i * dc_r);
                double newUR = (uR * conjR - uI * conjI);
                uI = (uR * conjI + uI * conjR);
                uR = newUR;
                newUR = uR / sqrt(uR * uR + uI * uI);
                uI = uI / sqrt(uR * uR + uI * uI);
                uR = newUR;
                double t = uR * vR + uI * vI + normLightHeight;
                t = t / (1. + normLightHeight);
                if (t < 0) {
                    t = 0;
                } else if (t > 1) {
                    t = 1;
                }
                double normShadowIntensity = 1 + (1 - normLightIntensity);
                tempR = (unsigned char) (max(0., min(255., tempR * normShadowIntensity)) +
                                         t * (max(0., min(255., tempR *
                                                                normLightIntensity)) -
                                              max(0., min(255., tempR *
                                                                normShadowIntensity))));
                tempG = (unsigned char) (max(0., min(255., tempG * normShadowIntensity)) +
                                         t * (max(0., min(255., tempG *
                                                                normLightIntensity)) -
                                              max(0., min(255., tempG *
                                                                normShadowIntensity))));
                tempB = (unsigned char) (max(0., min(255., tempB * normShadowIntensity)) +
                                         t * (max(0., min(255., tempB *
                                                                normLightIntensity)) -
                                              max(0., min(255., tempB *
                                                                normShadowIntensity))));
            }
            //endregion

            //region Stripe Average Colouring
            if (stripeLightIntensity != 1) {
                double lastOrbit = 0.5 + 0.5 * sin(stripeDensity * atan2(last_z_i, last_z_r));
                double smallCount = orbitCount - lastOrbit;
                orbitCount /= (double) i;
                smallCount /= (double) i - 1;
                double frac = -1. + log10(2.0 * log(R * R)) / log10(2.) -
                              log10(0.5 * log(last_z_r * last_z_r + last_z_i * last_z_i)) / log10(2.);
                double mix = frac * orbitCount + (1 - frac) * smallCount;
                if (mix < 0) {
                    mix = 0;
                } else if (mix > 1) {
                    mix = 1;
                }
                double stripeShadowIntensity = 1 + (1 - stripeLightIntensity);
                unsigned char stripeLightR = max(0., min(255., tempR * stripeLightIntensity));
                unsigned char stripeLightG = max(0., min(255., tempG * stripeLightIntensity));
                unsigned char stripeLightB = max(0., min(255., tempB * stripeLightIntensity));
                unsigned char stripeShadowR = max(0., min(255., tempR * stripeShadowIntensity));
                unsigned char stripeShadowG = max(0., min(255., tempG * stripeShadowIntensity));
                unsigned char stripeShadowB = max(0., min(255., tempB * stripeShadowIntensity));
                tempR = (unsigned char) (stripeShadowR + (mix * (stripeLightR - stripeShadowR)));
                tempG = (unsigned char) (stripeShadowG + (mix * (stripeLightG - stripeShadowG)));
                tempB = (unsigned char) (stripeShadowB + (mix * (stripeLightB - stripeShadowB)));
            }
            //endregion

            //region Border
            if (borderThickness > 0) {
                double tBorder = d / borderThickness;
                if (tBorder < 1) { // Border
                    tempR = (unsigned char) (borderR + tBorder * (tempR - borderR));
                    tempG = (unsigned char) (borderG + tBorder * (tempG - borderG));
                    tempB = (unsigned char) (borderB + tBorder * (tempB - borderB));
                }
            }
            //endregion

            image[currentIndex * 4] = tempR;
            image[currentIndex * 4 + 1] = tempG;
            image[currentIndex * 4 + 2] = tempB;
        }
    }
}

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
        double zoom, double posX, double posY) {
    cudaProfilerStart();

    //region Setup
    cout << "Setting up..." << endl;
    double ratio = (double) width / height;
    unsigned int size = width * height;

    unsigned char *imageHost;
    imageHost = (unsigned char *) malloc(4 * size * sizeof(unsigned char));
    unsigned char *imageDevice;
    gpuErrchk(cudaMallocManaged(&imageDevice, 4 * size * sizeof(unsigned char)));

    int suggestedBlockSize;
    int minGridSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &suggestedBlockSize, multibrot_kernel, 0, 4 * size);
    cout << "Suggested BlockSize: " << suggestedBlockSize << endl << "Min GridSize: " << minGridSize << endl;

    int gridSize = (size + blockSize - 1) / blockSize / unroll;
    cout << "BlockSize: " << blockSize << endl << "GridSize: " << gridSize << endl << "Unroll: " << unroll << endl;
    cout << "Setup done!" << endl;
    //endregion

    //region Generation
    cout << "Fractal generation in process..." << endl;
    multibrot_kernel<<<gridSize, blockSize>>>(unroll,
                                              imageDevice,
                                              width, height, ratio,
                                              exponent, iterations, R, eps,
                                              borderR, borderG, borderB, borderThickness,
                                              normOrbitSkip, normLightIntensity, normLightAngle, normLightHeight,
                                              bgR, bgG, bgB,
                                              kR, kG, kB, kD,
                                              internalBorderR, internalBorderG, internalBorderB,
                                              internalCoreR, internalCoreG, internalCoreB, internalK,
                                              stripeDensity, stripeLightIntensity,
                                              zoom, posX, posY);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaMemcpy(imageHost, imageDevice, 4 * size * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    cout << "Generation done!" << endl;

    int maxActiveBlocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, multibrot_kernel, blockSize, 0);
    int device;
    cudaDeviceProp props;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&props, device);
    double occupancy = (maxActiveBlocks * blockSize / props.warpSize) /
                       (double) (props.maxThreadsPerMultiProcessor / props.warpSize);
    cout << std::setprecision(4) << "Theoretical occupancy: " << occupancy << "%" << endl;
    //endregion

    for (int i = 0; i < size; i++) {
        rgb[i * 3 + 2] = imageHost[i * 4];
        rgb[i * 3 + 1] = imageHost[i * 4 + 1];
        rgb[i * 3] = imageHost[i * 4 + 2];
    }

    //region Cleanup
    free(imageHost);
    cudaFree(imageDevice);
    cudaDeviceReset();
    //endregion
    cudaProfilerStop();
}
