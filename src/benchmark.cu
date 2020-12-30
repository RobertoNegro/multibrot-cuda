#include "./cuda_kernel.cu"
#include <chrono>

int main(int argc, char **argv) {
    cudaProfilerStart();

    int width = 4000;
    int height = 2000;
    int blockSize = 32;
    int unroll = 4;
    int iterations = 10000;

    if (argc > 1) {
        int i = 1;
        while (i < argc) {
            string arg = argv[i++];
            if (i < argc) {
                if (arg == "-blocksize") {
                    blockSize = stoi(argv[i++]);
                } else if (arg == "-unroll") {
                    unroll = stoi(argv[i++]);
                } else if (arg == "-i") {
                    iterations = stoi(argv[i++]);
                }
            }
        }
    }

    double ratio = (double) width / height;
    unsigned int size = width * height;

    unsigned char *imageHost;
    imageHost = (unsigned char *) malloc(4 * size * sizeof(unsigned char));
    unsigned char *imageDevice;
    gpuErrchk(cudaMallocManaged(&imageDevice, 4 * size * sizeof(unsigned char)));

    int gridSize = (size + blockSize - 1) / blockSize / unroll;
    cout << "BlockSize: " << blockSize << endl << "GridSize: " << gridSize << endl << "Unroll: " << unroll << endl;

    chrono::steady_clock::time_point startTime = chrono::high_resolution_clock::now();
    multibrot_kernel<<<gridSize, blockSize>>>(unroll,
                                              imageDevice,
                                              width, height, ratio,
                                              2, iterations, 1000, 0,
                                              0, 0, 0, 0,
                                              3, 1, 0, 0,
                                              0, 0, 0,
                                              0, 0, 0,
                                              0, 0, 0, 255, 255, 255,
                                              0,
                                              0, 1,
                                              0.5, -0.6, 0);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaMemcpy(imageHost, imageDevice, 4 * size * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    chrono::steady_clock::time_point endTime = chrono::high_resolution_clock::now();
    double executionTime = chrono::duration<double>(endTime - startTime).count();
    printf("Time of generation: %.8fs\n", executionTime);

    free(imageHost);
    cudaFree(imageDevice);
    cudaDeviceReset();

    cudaProfilerStop();

    ofstream outfile;
    outfile.open("benchmark.txt", ios::out | ios::app);
    outfile << gridSize << "\t" << blockSize << "\t" << unroll << "\t" << iterations << "\t>>\t" << setprecision(8) << executionTime << "s" << endl;
    outfile.close();
}