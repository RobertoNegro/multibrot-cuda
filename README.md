# Multibrot CUDA
### Advanced Computing Architectures 2020/21 (UniTN)
##### Roberto Negro

Requires CUDA and OpenCV4 installed and correctly linked.
Tested on Mac OS 10.13.6 (High Sierra) with a GTX 1070 Ti (CUDA Toolkit 10.1 Update 1).

In Makefile, change CUDA_ROOT_DIR with your root directory of CUDA, and in CC_FLAGS change the path pointing to opencv4 as needed. 
Then, by executing the command ```make``` in the root of the project, the executable Multibrot should be created.
