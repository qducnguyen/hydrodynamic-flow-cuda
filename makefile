EXECS?=incompressible_flow
CC?=gcc
NVCC?=nvcc

all: ${EXECS}
incompressible_flow: incompressible_flow.c
	${CC} -o incompressible_flow incompressible_flow.c
incompressible_flow_cuda: incompressible_flow_cuda.cu
	${NVCC} -o incompressible_flow_cuda incompressible_flow_cuda.cu
clean:
	rm ${EXECS}