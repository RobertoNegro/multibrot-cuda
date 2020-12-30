###########################################################
## USER SPECIFIC DIRECTORIES ##
CUDA_ROOT_DIR=/usr/local/cuda

##########################################################
## CC COMPILER OPTIONS ##
CC=g++
CC_FLAGS= -O3 -std=c++11 -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_highgui -I/usr/local/include/opencv4
CC_LIBS=

##########################################################
## NVCC COMPILER OPTIONS ##
NVCC=nvcc
NVCC_FLAGS=
NVCC_LIBS=

CUDA_LIB_DIR= -L$(CUDA_ROOT_DIR)/lib
CUDA_INC_DIR= -I$(CUDA_ROOT_DIR)/include
CUDA_LINK_LIBS= -lcudart

##########################################################
## Project file structure ##
SRC_DIR = src
OBJ_DIR = bin
INC_DIR = include

##########################################################
## Make variables ##
EXE = Multibrot
OBJS = $(OBJ_DIR)/main.o $(OBJ_DIR)/cuda_kernel.o

##########################################################
## Compile ##

# Link c++ and CUDA compiled object files to target executable:
$(EXE) : $(OBJS)
	$(CC) $(CC_FLAGS) $(OBJS) -o $@ $(CUDA_INC_DIR) $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS)
# Compile main .cpp file to object files:
$(OBJ_DIR)/%.o : %.cpp
	$(CC) $(CC_FLAGS) -c $< -o $@
# Compile C++ source files to object files:
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cpp include/%.h
	$(CC) $(CC_FLAGS) -c $< -o $@
# Compile CUDA source files to object files:
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cu $(INC_DIR)/%.cuh
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(NVCC_LIBS)
# Clean objects in object directory.
clean:
	$(RM) bin/* *.o $(EXE)

