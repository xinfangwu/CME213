# Google test installation directories.
# Make sure these directories exist.
GTEST_ROOT = /home/cme213/xinfangw/CME213/CodeCUDA/googletest-main
GTEST_DIR = $(GTEST_ROOT)/googletest
GTEST_INC = $(GTEST_ROOT)/googletest/include/

CXX=nvcc
CXXFLAGS=-O2 --std=c++14 -Icuda_helper -I$(GTEST_INC) --gpu-architecture=compute_75 --gpu-code=sm_75 --generate-line-info

# Linker
LDFLAGS := gtest.a

SRC := $(wildcard *.cpp) $(wildcard *.cu)
EXE := $(patsubst %.cpp,%,$(filter %.cpp,$(SRC))) $(patsubst %.cu,%,$(filter %.cu,$(SRC)))

.PHONY: all clean gtest

all: gtest $(EXE)

# pattern rules
% : %.cu
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $<

% : %.cpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $<

clean:
	$(RM) $(EXE) *.o *.a

GTEST_FLAGS=-O2 --std=c++14 -isystem $(GTEST_INC)

# Google Test headers
GTEST_HEADERS = $(GTEST_INC)/gtest/*.h \
                $(GTEST_INC)/gtest/internal/*.h

# Builds gtest.a and gtest_main.a.
GTEST_SRCS_ = $(GTEST_DIR)/src/*.cc $(GTEST_DIR)/src/*.h $(GTEST_HEADERS)

gtest: gtest.a gtest_main.a

gtest-all.o : $(GTEST_SRCS_)
	$(CXX) $(GTEST_FLAGS) -I$(GTEST_DIR) -c \
    	$(GTEST_DIR)/src/gtest-all.cc

gtest_main.o : $(GTEST_SRCS_)
	$(CXX) $(GTEST_FLAGS) -I$(GTEST_DIR) -c \
		$(GTEST_DIR)/src/gtest_main.cc
            
gtest.a : gtest-all.o
	$(AR) $(ARFLAGS) $@ $^

gtest_main.a : gtest-all.o gtest_main.o
	$(AR) $(ARFLAGS) $@ $^
