CC := g++

EXEC := hello_world_opencl_c-api  hello_world_opencl_c++  mult_mat_vect

INC := -I/usr/local/cuda/include

LDFLAGS := -L/usr/lo-lOpenCL -Xlinker -rpath=".:"

ifeq ($(DEBUG),yes)
	CFLAGS := -g -pg -Wall -Wno-comment -DDEBUG
else
	CFLAGS := -O2 -Wall -Wno-comment 
endif


all: $(EXEC)


mult_mat_vect: ../src/mult_mat_vect.cpp ../src/mult_mat_vect_opencl.cpp
	$(CC) -o $@ $(CFLAGS) $(INC) ../src/mult_mat_vect.cpp ../src/mult_mat_vect_opencl.cpp ../../code/build/libcommon.so $(LDFLAGS) 

%: ../src/%.cpp
	$(CC) -o $@ $(CFLAGS) $(INC) "../src/$@.cpp" $(LDFLAGS) 
	
libcommon.so: ../src/common.cpp ../src/tools.cpp
	$(CC) -o $@ -shared -fPIC $(CFLAGS) $(INC) $+ $(LDFLAGS) 

.PHONY: clean mrproper tags

clean:
	rm -f $(EXEC)

help:
	@echo
	@echo "make  -> build target in release mode"
	@echo "make DEBUG=yes  -> build target in debug mode"
	@echo "make test  -> to debug Makefile"
	@echo

test:
	@echo CC=$(CC) 
	@echo EXEC=$(EXEC)
	@echo INC=$(INC) 
	@echo LDFLAGS=$(LDFLAGS) 

