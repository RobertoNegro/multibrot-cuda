#include "./cuda_kernel.cpp"
#include <chrono>
#include <vector>
#include <thread>

int main(int argc, char **argv) {
    int width = 4000;
    int height = 2000;
    int unroll = 1;
    int iterations = 10000;
    int nThreads = 12;
    double eps = 0;

    if (argc > 1) {
        int i = 1;
        while (i < argc) {
            string arg = argv[i++];
            if (i < argc) {
                if (arg == "-unroll")
                    unroll = stoi(argv[i++]);
                else if (arg == "-i")
                    iterations = stoi(argv[i++]);
                else if (arg == "-threads")
                    nThreads = stoi(argv[i++]);
                else if (arg == "-eps")
                    eps = stod(argv[i++]);
            }
        }
    }

    double ratio = (double) width / height;
    unsigned int size = width * height;

    unsigned char *imageHost;
    imageHost = (unsigned char *) malloc(4 * size * sizeof(unsigned char));

    cout << "Unroll: " << unroll << endl << "Eps: " << eps << endl;

    chrono::steady_clock::time_point startTime = chrono::high_resolution_clock::now();
    vector<thread> threads;
    for (int i = 0; i < nThreads; i++) {
        threads.push_back(thread(multibrot_kernel,
                                 i,
                                 ceil((double)size / nThreads),
                                 imageHost,
                                 width, height, ratio,
                                 2, iterations, 1000, eps,
                                 0, 0, 0, 0,
                                 3, 1, 0, 0,
                                 0, 0, 0,
                                 0, 0, 0, 0,
                                 0, 0, 0, 255, 255, 255,
                                 0,
                                 0, 1,
                                 0.5, -0.6, 0));
    }
    for (thread &th : threads) {
        th.join();
    }

    chrono::steady_clock::time_point endTime = chrono::high_resolution_clock::now();
    double executionTime = chrono::duration<double>(endTime - startTime).count();
    printf("Time of generation: %.8fs\n", executionTime);

    free(imageHost);

    ofstream outfile;
    outfile.open("benchmark.txt", ios::out | ios::app);
    outfile << "CPU" << "\t" << nThreads << "\t" << unroll << "\t" << iterations << "\t" << setprecision(16) << eps
            << "\t>>\t" << setprecision(8)
            << executionTime << "s" << endl;
    outfile.close();
}