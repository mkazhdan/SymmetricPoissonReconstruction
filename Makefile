SYMMETRIC_POISSON_RECON_TARGET=SymmetricPoissonRecon
SYMMETRIC_POISSON_RECON_SOURCE=SymmetricPoissonRecon/SymmetricPoissonRecon.cpp

COMPILER ?= gcc
#COMPILER ?= clang

ifeq ($(COMPILER),gcc)
	CFLAGS += -fopenmp -Wno-deprecated -std=c++17 -Wno-invalid-offsetof
	LFLAGS += -lgomp -lstdc++ -lpthread
	CC=gcc
	CXX=g++
else
	CFLAGS += -Wno-deprecated -std=c++17 -Wno-invalid-offsetof -Wno-dangling-else -Wno-null-dereference
	LFLAGS += -lstdc++
	CC=clang
	CXX=clang++
endif

CFLAGS += -O3 -DRELEASE -funroll-loops -g
LFLAGS += -O3 -g

BIN = Bin/Linux/
BIN_O = Obj/Linux/
INCLUDE = . -I/mnt/c/Users/mkazh/OneDrive\ -\ Johns\ Hopkins/Research/Libraries/Include


MD=mkdir

SYMMETRIC_POISSON_RECON_OBJECTS=$(addprefix $(BIN_O), $(addsuffix .o, $(basename $(SYMMETRIC_POISSON_RECON_SOURCE))))
SYMMETRIC_POISSON_RECON_OBJECT_DIR=$(dir $(SYMMETRIC_POISSON_RECON_OBJECTS))

all: make_dirs
all: $(BIN)$(SYMMETRIC_POISSON_RECON_TARGET)

symmetricpoissonrecon: make_dirs
symmetricpoissonrecon: $(BIN)$(SYMMETRIC_POISSON_RECON_TARGET)

clean:
	rm -rf $(BIN)$(SYMMETRIC_POISSON_RECON_TARGET)
	rm -rf $(BIN_O)

make_dirs: FORCE
	$(MD) -p $(BIN)
	$(MD) -p $(BIN_O)
	$(MD) -p $(SYMMETRIC_POISSON_RECON_OBJECT_DIR)

$(BIN)$(SYMMETRIC_POISSON_RECON_TARGET): $(SYMMETRIC_POISSON_RECON_OBJECTS)
	$(CXX) -o $@ $(SYMMETRIC_POISSON_RECON_OBJECTS) -L$(BIN) $(LFLAGS)

$(BIN_O)%.o: $(SRC)%.cpp
	$(CXX) -c -o $@ $(CFLAGS) -I$(INCLUDE) $<

FORCE: