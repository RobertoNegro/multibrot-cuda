#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <vector>
#include <random>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <chrono>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "include/cuda_kernel.cuh"

using namespace std;
using namespace cv;

int main(int argc, char **argv) {
    //region Parameters
    int width = 1840;
    int height = 1000;

    int exponent = 2;
    int iterations = 100;
    double R = 1000.;
    double eps = 0.0001;

    unsigned char bgR = 0x26;
    unsigned char bgG = 0x46;
    unsigned char bgB = 0x53;
    double kR = 0.;
    double kG = 0.;
    double kB = 0.;

    unsigned char internalBorderR = 0xf4;
    unsigned char internalBorderG = 0xa2;
    unsigned char internalBorderB = 0x61;
    unsigned char internalCoreR = 0xe7;
    unsigned char internalCoreG = 0x6f;
    unsigned char internalCoreB = 0x51;
    double internalK = 10.;

    double zoom = 0.25;
    double posX = -0.6;
    double posY = 0;

//    double zoom = 45;
//    double posX = -0.761574;
//    double posY = -0.0847596;

//    double zoom = 100.5;
//    double posX = -0.55;
//    double posY = -0.5;

    unsigned char borderR = 0xe9;
    unsigned char borderG = 0xc4;
    unsigned char borderB = 0x6a;
    double borderThickness = 0.00001;

    long normOrbitSkip = 3;
    double stripeDensity = 3.5;
    double normLightIntensity = 1.5;
    double normLightAngle = -45.;
    double normLightHeight = 1.5;

    double stripeLightIntensity = 1.3;

    bool saveOnly = false;
    int blockSize = 32;
    int unroll = 1;

    if (argc > 1) {
        int i = 1;
        while (i < argc) {
            string arg = argv[i++];
            if (i < argc) {
                if (arg == "-w")
                    width = stoi(argv[i++]);
                else if (arg == "-h")
                    height = stoi(argv[i++]);
                else if (arg == "-i")
                    iterations = stoi(argv[i++]);
                else if (arg == "-z")
                    zoom = stod(argv[i++]);
                else if (arg == "-x")
                    posX = stod(argv[i++]);
                else if (arg == "-y")
                    posY = stod(argv[i++]);
                else if (arg == "-R")
                    R = stod(argv[i++]);
                else if (arg == "-eps")
                    eps = stod(argv[i++]);
                else if (arg == "-kR")
                    kR = stod(argv[i++]);
                else if (arg == "-kG")
                    kG = stod(argv[i++]);
                else if (arg == "-kB")
                    kB = stod(argv[i++]);
                else if (arg == "-kI")
                    internalK = stod(argv[i++]);
                else if (arg == "-sI")
                    stripeLightIntensity = stod(argv[i++]);
                else if (arg == "-sD")
                    stripeDensity = stod(argv[i++]);
                else if (arg == "-nI")
                    normLightIntensity = stod(argv[i++]);
                else if (arg == "-nH")
                    normLightHeight = stod(argv[i++]);
                else if (arg == "-nA")
                    normLightAngle = stod(argv[i++]);
                else if (arg == "-bC") {
                    char *hex = argv[i++];
                    char subhex[3];
                    strncpy(subhex, &hex[0], 2);
                    subhex[2] = '\0';
                    borderR = stoi(subhex, 0, 16);
                    strncpy(subhex, &hex[2], 2);
                    subhex[2] = '\0';
                    borderG = stoi(subhex, 0, 16);
                    strncpy(subhex, &hex[4], 2);
                    subhex[2] = '\0';
                    borderB = stoi(subhex, 0, 16);
                } else if (arg == "-bT") {
                    borderThickness = stod(argv[i++]);
                } else if (arg == "-bg") {
                    char *hex = argv[i++];
                    char subhex[3];
                    strncpy(subhex, &hex[0], 2);
                    subhex[2] = '\0';
                    bgR = stoi(subhex, 0, 16);
                    strncpy(subhex, &hex[2], 2);
                    subhex[2] = '\0';
                    bgG = stoi(subhex, 0, 16);
                    strncpy(subhex, &hex[4], 2);
                    subhex[2] = '\0';
                    bgB = stoi(subhex, 0, 16);
                } else if (arg == "-ib") {
                    char *hex = argv[i++];
                    char subhex[3];
                    strncpy(subhex, &hex[0], 2);
                    subhex[2] = '\0';
                    internalBorderR = stoi(subhex, 0, 16);
                    strncpy(subhex, &hex[2], 2);
                    subhex[2] = '\0';
                    internalBorderG = stoi(subhex, 0, 16);
                    strncpy(subhex, &hex[4], 2);
                    subhex[2] = '\0';
                    internalBorderB = stoi(subhex, 0, 16);
                } else if (arg == "-ic") {
                    char *hex = argv[i++];
                    char subhex[3];
                    strncpy(subhex, &hex[0], 2);
                    subhex[2] = '\0';
                    internalCoreR = stoi(subhex, 0, 16);
                    strncpy(subhex, &hex[2], 2);
                    subhex[2] = '\0';
                    internalCoreG = stoi(subhex, 0, 16);
                    strncpy(subhex, &hex[4], 2);
                    subhex[2] = '\0';
                    internalCoreB = stoi(subhex, 0, 16);
                } else if (arg == "-save") {
                    saveOnly = true;
                } else if (arg == "-blocksize") {
                    blockSize = stoi(argv[i++]);
                } else if (arg == "-unroll") {
                    unroll = stoi(argv[i++]);
                }
            }
        }
    }
    //endregion

    if (!saveOnly)
        namedWindow("Multibrot", WINDOW_OPENGL);

    width = width * 2;
    height = height * 2;

    unsigned char internalBorderROrig = internalBorderR;
    unsigned char internalBorderGOrig = internalBorderG;
    unsigned char internalBorderBOrig = internalBorderB;
    unsigned char internalCoreROrig = internalCoreR;
    unsigned char internalCoreGOrig = internalCoreG;
    unsigned char internalCoreBOrig = internalCoreB;

    unsigned char *rgb = nullptr;
    bool hasToExit = false;
    while (!hasToExit) {
        auto startTime = chrono::high_resolution_clock::now();
        rgb = (unsigned char *) malloc(width * height * 3 * sizeof(unsigned char));
        multibrot(
                unroll,
                blockSize,
                rgb,
                width, height,
                exponent, iterations, R, eps,
                borderR, borderG, borderB, borderThickness * (1. / zoom),
                normOrbitSkip, normLightIntensity, normLightAngle, normLightHeight,
                bgR, bgG, bgB,
                kR, kG, kB,
                internalBorderR, internalBorderG, internalBorderB, internalCoreR, internalCoreG, internalCoreB,
                internalK,
                stripeDensity, stripeLightIntensity,
                zoom, posX, posY);
        auto endTime = chrono::high_resolution_clock::now();
        double executionTime = chrono::duration<double>(endTime - startTime).count();
        //region Image showing


        printf("---------------------\n");
        printf("Background color (z): %02x%02x%02x\n", bgR, bgG, bgB);
        printf("Internal border color (z): %02x%02x%02x\n", internalBorderR, internalBorderG, internalBorderB);
        printf("Internal core color (z): %02x%02x%02x\n", internalCoreR, internalCoreG, internalCoreB);
        printf("Border color (z): %02x%02x%02x\n", borderR, borderG, borderB);
        printf("Border thickness (3 | 4): %.8f\n", borderThickness);
        printf("Position: X (a | d): %.8f ; Y (w | s): %.8f\n", posX, posY);
        printf("Zoom (+ | -): %.8f\n", zoom);
        printf("Epsilon (o | p): %.16f\n", eps);
        printf("Iterations (k | l): %d\n", iterations);
        printf("R: %.4f\n", R);
        printf("External K: R (u | i): %.8f ; G (h | j): %.8f ; B (v | b): %.8f\n", kR, kG, kB);
        printf("Internal K (n | m): %.8f\n", internalK);
        printf("Stripe light intensity (r | e): %.8f\n", stripeLightIntensity);
        printf("Stripe density (x | c): %.8f\n", stripeDensity);
        printf("Norm light intensity (1 | 2): %.8f\n", normLightIntensity);
        printf("Norm light height (f | g): %.8f\n", normLightHeight);
        printf("Norm light angle (t | y): %.8f\n", normLightAngle);

        printf("Time of generation: %.8fs\n", executionTime);

        char settings[1024];
        sprintf(settings,
                "./Multibrot -w %d -h %d -bg %02x%02x%02x -ib %02x%02x%02x -ic %02x%02x%02x -bC %02x%02x%02x -bT %.8f -x %.8f -y %.8f -z %.8f -eps %.16f -i %d -R %.4f -kR %.8f -kG %.8f -kB %.8f -kI %.8f -sI %.8f -sD %.8f -nI %.8f -nH %.8f -nA %.8f\n",
                width / 2, height / 2,
                bgR, bgG, bgB,
                internalBorderR, internalBorderG, internalBorderB,
                internalCoreR, internalCoreG, internalCoreB,
                borderR, borderG, borderB, borderThickness,
                posX, posY,
                zoom,
                eps,
                iterations,
                R,
                kR, kG, kB,
                internalK,
                stripeLightIntensity,
                stripeDensity,
                normLightIntensity,
                normLightHeight,
                normLightAngle);
        printf("%s", settings);

        if (!saveOnly)
            printf("Press ESC to exit\n");

        Mat mat = Mat(height, width, CV_8UC3, rgb);
        resize(mat, mat, Size(width / 2, height / 2), 0, 0, INTER_AREA);

        if (!saveOnly) {
            imshow("Multibrot", mat);

            bool validKey = false;
            while (!validKey) {
                unsigned char key = waitKey(0) & 0xFF;
                printf("Pressed key %d\n", key);

                validKey = true;
                switch (key) {
                    case 27: {
                        // ESC
                        hasToExit = true;
                    }
                        break;
                    case 119: {
                        // w
                        posY -= 0.1 * (1. / zoom);
                    }
                        break;
                    case 115: {
                        // s
                        posY += 0.1 * (1. / zoom);
                    }
                        break;
                    case 97: {
                        // a
                        posX -= 0.1 * (1. / zoom);
                    }
                        break;
                    case 100: {
                        // d
                        posX += 0.1 * (1. / zoom);
                    }
                        break;
                    case 43: {
                        // +
                        zoom += 0.1 * (zoom);
                    }
                        break;
                    case 45: {
                        // -
                        zoom -= 0.1 * (zoom);
                    }
                        break;
                    case 32: {
                        // space
                        time_t now = time(0);
                        tm *ltm = localtime(&now);
                        stringstream fileName;
                        fileName << "result_" << put_time(ltm, "%Y-%m-%d_%H-%M-%S") << ".jpg";
                        imwrite(fileName.str(), mat);

                        ofstream outfile;
                        outfile.open("results.txt", ios::out | ios::app);
                        outfile << fileName.str().c_str() << "\t" << setprecision(8) << executionTime << "s" << "\t"
                                << settings << endl;
                        outfile.close();

                        printf("File saved as: %s\n", fileName.str().c_str());
                    }
                        break;
                    case 111: {
                        // o
                        if (eps > 0.0000000000000001) {
                            eps /= 10.;
                        }
                    }
                        break;
                    case 112: {
                        // p
                        if (eps == 0) {
                            eps = 0.0000000000000001;
                        } else if (eps < 1) {
                            eps *= 10.;
                        }
                    }
                        break;
                    case 107: {
                        // k
                        if (iterations > 0) {
                            iterations /= 10.;
                        }
                    }
                        break;
                    case 108: {
                        // l
                        if (iterations == 0) {
                            iterations = 10.;
                        } else if (iterations < 1000000) {
                            iterations *= 10.;
                        }
                    }
                    case 110: {
                        // n
                        if (internalK > 0) {
                            internalK -= 1.;
                        }
                    }
                        break;
                    case 109: {
                        // m
                        internalK += 1.;
                    }
                        break;
                    case 117: {
                        // u
                        if (kR > 0) {
                            kR -= 0.5;
                        }
                    }
                        break;
                    case 105: {
                        // i
                        kR += 0.5;
                    }
                        break;
                    case 104: {
                        // h
                        if (kG > 0) {
                            kG -= 0.5;
                        }
                    }
                        break;
                    case 106: {
                        // j
                        kG += 0.5;
                    }
                        break;
                    case 118: {
                        // v
                        if (kB > 0) {
                            kB -= 0.5;
                        }
                    }
                        break;
                    case 98: {
                        // b
                        kB += 0.5;
                    }
                        break;
                    case 116: {
                        // t
                        normLightAngle -= 45;
                    }
                        break;
                    case 121: {
                        // y
                        normLightAngle += 45;
                    }
                        break;
                    case 102: {
                        // f
                        if (normLightHeight > 0) {
                            normLightHeight -= 0.5;
                        }
                    }
                        break;
                    case 103: {
                        // g
                        normLightHeight += 0.5;
                    }
                        break;
                    case 120: {
                        // x
                        if (stripeDensity > 0) {
                            stripeDensity -= 0.1;
                        }
                    }
                        break;
                    case 99: {
                        // c
                        stripeDensity += 0.1;
                    }
                        break;
                    case 101: {
                        // e
                        if (stripeLightIntensity > 0) {
                            stripeLightIntensity -= 0.1;
                        }
                    }
                        break;
                    case 114: {
                        // r
                        if (stripeLightIntensity < 2) {
                            stripeLightIntensity += 0.1;
                        }
                    }
                        break;
                    case 49: {
                        // 1
                        if (normLightIntensity > 0) {
                            normLightIntensity -= 0.1;
                        }
                    }
                        break;
                    case 50: {
                        // 2
                        if (normLightIntensity < 2) {
                            normLightIntensity += 0.1;
                        }
                    }
                        break;
                    case 51: {
                        // 3
                        if (borderThickness > 0.0000001) {
                            borderThickness /= 10.;
                        }
                    }
                        break;
                    case 52: {
                        // 4
                        if (borderThickness == 0) {
                            borderThickness = 10.;
                        } else if (borderThickness < 1) {
                            borderThickness *= 10.;
                        }
                    }
                        break;
                    case 122: {
                        // z
                        if (internalBorderR == 0 && internalBorderG == 0 && internalBorderB == 0 &&
                            internalCoreR == 0 &&
                            internalCoreG == 0 && internalCoreB == 0) {
                            internalBorderR = 255;
                            internalBorderG = 255;
                            internalBorderB = 255;
                            internalCoreR = 255;
                            internalCoreG = 255;
                            internalCoreB = 255;
                        } else if (internalBorderR == 255 && internalBorderG == 255
                                   && internalBorderB == 255 && internalCoreR == 255
                                   && internalCoreG == 255 && internalCoreB == 255) {
                            internalBorderR = internalBorderROrig;
                            internalBorderG = internalBorderGOrig;
                            internalBorderB = internalBorderBOrig;
                            internalCoreR = internalCoreROrig;
                            internalCoreG = internalCoreGOrig;
                            internalCoreB = internalCoreBOrig;
                        } else {
                            internalBorderR = 0;
                            internalBorderG = 0;
                            internalBorderB = 0;
                            internalCoreR = 0;
                            internalCoreG = 0;
                            internalCoreB = 0;
                        }
                    }
                        break;
                    default: {
                        validKey = false;
                    }
                        break;
                }
            }
        } else {
            time_t now = time(0);
            tm *ltm = localtime(&now);
            stringstream fileName;
            fileName << "result_" << put_time(ltm, "%Y-%m-%d_%H-%M-%S") << ".jpg";
            imwrite(fileName.str(), mat);

            ofstream outfile;
            outfile.open("results.txt", ios::out | ios::app);
            outfile << fileName.str().c_str() << "\t" << setprecision(8) << executionTime << "s" << "\t"
                    << settings << endl;
            outfile.close();

            printf("File saved as: %s\n", fileName.str().c_str());
        }
    }

    //endregion

    if(rgb != nullptr) {
        free(rgb);
    }

//    imwrite("result.jpg", mat);

//    //region Save file
//    cout << "Saving file.." << endl;
//    saveFile(&filename, width, height, r, g, b);
//    cout << "Converting file to jpeg.." << endl;
//    system(("magick " + filename + ".ppm " + filename + ".jpg").c_str());
//    cout << "Save completed!" << endl;
//    //endregion

    return 0;
}

