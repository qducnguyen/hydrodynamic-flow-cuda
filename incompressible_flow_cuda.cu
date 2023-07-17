#include <stdio.h>
#include <malloc.h>
#include <string.h>
#include <cuda.h>
#include <time.h>
#include <sys/time.h>


#define nx 32
#define ny 32
#define nt 10000
#define nit 50 
#define c 1.0
#define xmax 2.0
#define ymax 2.0 
#define rho 1.0 
#define nu 0.1
#define dt 0.001

#define GridSizeX 2
#define GridSizeY 2
#define BlockSizeX nx / GridSizeX
#define BlockSizeY ny / GridSizeY

void save_results(double *u, double *v, double *p, char *filename, double dx, double dy){
	//  
	FILE *file = fopen(filename, "w");
	int i,j;

	fprintf(file, "%d\n", nx);
	fprintf(file, "%d\n", ny);
	fprintf(file, "%d\n", nt);
	fprintf(file, "%d\n", nit);
	fprintf(file, "%f\n", c);
	fprintf(file, "%f\n", xmax);
	fprintf(file, "%f\n", ymax);
	fprintf(file, "%f\n", rho);
	fprintf(file, "%f\n", nu);
	fprintf(file, "%f\n", dt);
	fprintf(file, "%f\n", dx);
	fprintf(file, "%f\n", dy);


	// for (i = 0; i < ny; i++){
	// 	for (j = 0; j < nx; j++){
	// 		// 73 pecision ..
	// 		fprintf(file, "%.73lf ", *(u + i*nx + j));
	// 	}
	// 	fprintf(file, "\n");
	// }

	// for (i = 0; i < ny; i++){
	// 	for (j = 0; j < nx; j++){
	// 		// 73 pecision ..
	// 		fprintf(file, "%.73lf ", *(v + i*nx + j));
	// 	}
	// 	fprintf(file, "\n");
	// }

	for (i = 0; i < ny; i++){
		for (j = 0; j < nx; j++){
			// 73 pecision ..
			fprintf(file, "%.73lf ", *(p + i*nx + j));
		}
		fprintf(file, "\n");
	}

	fclose(file);
}

void print_array(double *arr){
	printf("\n\n");	
	int i, j;
	for (i = 0; i < ny; i++){
		for (j = 0; j < nx; j++){
			printf("%e ", *(arr + i * nx + j));
		}
		printf("\n\n");
	}
}


void init(double *u, double *v, double *p, double *pn, double *b){
	int i, j;
	for (i = 0; i < ny; i++){
		for (j = 0; j < nx; j++){
			*(u + i * nx + j) = 0.0;
			*(v + i * nx + j) = 0.0;
			*(p + i * nx + j) = 0.0;
			*(pn + i * nx + j) = 0.0;
			*(b + i * nx + j) = 0.0;
		}
	}
}



__global__ void build_up_b(double *b, double *u, double *v, double dx, double dy){

	__shared__ double tile_u[BlockSizeY+2][BlockSizeX+2];
	__shared__ double tile_v[BlockSizeY+2][BlockSizeX+2];

	int ty = threadIdx.y;
    int tx = threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;	


    if (i < ny && j < nx) {
    	tile_u[ty+1][tx+1] = *(u +i*nx+j);
		tile_v[ty+1][tx+1] = *(v +i*nx+j);
    }
    if (ty == 0 && i > 0) {
        tile_u[ty][tx+1] = *(u+(i-1)*nx+j); 
        tile_v[ty][tx+1] = *(v+(i-1)*nx+j); 

    }
    if (ty == BlockSizeY-1 && i < ny-1) {
        tile_u[ty+2][tx+1] = *(u + (i+1)*nx +j);
        tile_v[ty+2][tx+1] = *(v + (i+1)*nx +j);

    }
    if (tx == 0 && j > 0) {
        tile_u[ty+1][tx] = *(u + i*nx+j-1);
        tile_v[ty+1][tx] = *(v + i*nx+j-1);

    }
    if (tx == BlockSizeX-1 && j < nx-1) {
        tile_u[ty+1][tx+2] = *(u + i*nx+j+1);
        tile_v[ty+1][tx+2] = *(v + i*nx+j+1);

    }

    __syncthreads();

    if (i > 0 && i < ny - 1 && j > 0 && j < nx - 1){
        *(b + i * nx + j) = rho * (1 / dt *
				((tile_u[ty+1][tx+2] - tile_u[ty+1][tx]) / (2*dx)
				+(tile_v[ty+2][tx+1] - tile_v[ty][tx+1]) / (2*dy)) -
			(tile_u[ty+1][tx+2] - tile_u[ty+1][tx]) * (tile_u[ty+1][tx+2]- tile_u[ty+1][tx]) / (2*2*dx*dx) -
			2 * ((tile_u[ty+2][tx+1] - tile_u[ty][tx+1]) / (2*dy) *
			(tile_v[ty+1][tx+2] - tile_v[ty+1][tx]) / (2*dx)) - 
			(tile_v[ty+2][tx+1] - tile_v[ty][tx+1]) * (tile_v[ty+2][tx+1] - tile_v[ty][tx+1]) / (2*2*dy*dy));
    }


}

__global__ void  solve_pressure_poisson(double *p, double *pn, double *b, double dx, double dy){

	__shared__ double tile[BlockSizeY+2][BlockSizeX+2];

	int ty = threadIdx.y;
    int tx = threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;	


   	if (i < ny && j < nx) {
    	tile[ty+1][tx+1] = *(pn +i*nx+j);
    }
    if (ty == 0 && i > 0) {
        tile[ty][tx+1] = *(pn+(i-1)*nx+j); 
    }
    if (ty == BlockSizeY-1 && i < ny-1) {
        tile[ty+2][tx+1] = *(pn + (i+1)*nx +j);
    }
    if (tx == 0 && j > 0) {
        tile[ty+1][tx] = *(pn + i*nx+j-1);
    }
    if (tx == BlockSizeX-1 && j < nx-1) {
        tile[ty+1][tx+2] = *(pn + i*nx+j+1);
    }

    __syncthreads();


    if (i > 0 && i < ny - 1 && j > 0 && j < nx - 1){
        double tmp = ((tile[ty+1][tx+2] +tile[ty+1][tx]) * dy*dy  + 
                        (tile[ty+2][tx+1] + tile[ty][tx+1]) * dx*dx)/
                        (2 * (dx*dx + dy*dy))- 
                        dx*dx*dy*dy / (2 * (dx*dx + dy*dy)) * *(b + i*nx +j);

        tile[ty+1][tx+1] = tmp;
    }
	// __syncthreads();


	if (j == nx-1){
        tile[ty+1][tx+1] = tile[ty+1][tx];
    }

    if (i == 0){
    	tile[ty+1][tx+1] = tile[ty+2][tx+1];	
    }

    if (j == 0){
        tile[ty+1][tx+1] = tile[ty+1][tx+2];
    }

    if (i == ny - 1){
        tile[ty+1][tx+1] = 0;
    }

	__syncthreads();

	if (i < ny && j < nx) {
		*(p + i*nx + j) = tile[ty+1][tx+1];
	}

	
}


__global__ void velocity_update(double *u, double *v, double *p, double dx, double dy){

	__shared__ double tile_u[BlockSizeY+2][BlockSizeX+2];
	__shared__ double tile_v[BlockSizeY+2][BlockSizeX+2];

	int ty = threadIdx.y;
    int tx = threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;	


    if (i < ny && j < nx) {
    	tile_u[ty+1][tx+1] = *(u +i*nx+j);
		tile_v[ty+1][tx+1] = *(v +i*nx+j);
    }
    if (ty == 0 && i > 0) {
        tile_u[ty][tx+1] = *(u+(i-1)*nx+j); 
        tile_v[ty][tx+1] = *(v+(i-1)*nx+j); 

    }
    if (ty == BlockSizeY-1 && i < ny-1) {
        tile_u[ty+2][tx+1] = *(u + (i+1)*nx +j);
        tile_v[ty+2][tx+1] = *(v + (i+1)*nx +j);

    }
    if (tx == 0 && j > 0) {
        tile_u[ty+1][tx] = *(u + i*nx+j-1);
        tile_v[ty+1][tx] = *(v + i*nx+j-1);

    }
    if (tx == BlockSizeX-1 && j < nx-1) {
        tile_u[ty+1][tx+2] = *(u + i*nx+j+1);
        tile_v[ty+1][tx+2] = *(v + i*nx+j+1);

    }

    __syncthreads();


    if (i > 0 && i < ny - 1 && j > 0 && j < nx - 1){

        double temp_u = tile_u[ty+1][tx+1] - tile_u[ty+1][tx+1] * dt / dx  * 
								(tile_u[ty+1][tx+1] - tile_u[ty+1][tx]) -
								tile_v[ty+1][tx+1] * dt / dy *
								(tile_u[ty+1][tx+1] - tile_u[ty][tx+1]) - 
								dt / (2 * rho * dx) * (*(p + i*nx + j+1) - *(p + i*nx +j-1)) +
								nu * (dt / (dx*dx) *
								(tile_u[ty+1][tx+2] - 2 * tile_u[ty+1][tx+1] + tile_u[ty+1][tx]) +
								dt / (dy*dy) *
								(tile_u[ty+2][tx+1] - 2 * tile_u[ty+1][tx+1] + tile_u[ty][tx+1]));

        double temp_v = tile_v[ty+1][tx+1] - tile_u[ty+1][tx+1] * dt / dx  * 
                                (tile_v[ty+1][tx+1] - tile_v[ty+1][tx]) -
                           		tile_v[ty+1][tx+1] * dt / dy *
                            	(tile_v[ty+1][tx+1] -tile_v[ty][tx+1]) - 
                            	dt / (2 * rho * dy) * (*(p + (i+1)*nx + j) - *(p + (i-1)*nx +j)) +
                            	nu * (dt / (dx*dx) *
                            	(tile_v[ty+1][tx+2] - 2 * tile_v[ty+1][tx+1] + tile_v[ty+1][tx]) +
                            	dt/ (dy*dy) *
                            	(tile_v[ty+2][tx+1] - 2 * tile_v[ty+1][tx+1] + tile_v[ty][tx+1]));

		tile_u[ty+1][tx+1] = temp_u;
		tile_v[ty+1][tx+1] = temp_v;
    }
    __syncthreads();


    if (j == 0){
       tile_u[ty+1][tx+1]= 0;
   	   tile_v[ty+1][tx+1] = 0;
    }
    
    if (j == nx - 1){
 		tile_u[ty+1][tx+1]= 0;
 		tile_v[ty+1][tx+1] = 0;
      
    }

    if (i == 0){
        tile_u[ty+1][tx+1] = 0;
 		tile_v[ty+1][tx+1] = 0;
    }

    if (i == ny - 1){
     	tile_u[ty+1][tx+1] = 1; 	// set velocity on cavity lid equal to 1	
		tile_v[ty+1][tx+1] = 0; 

    }


    __syncthreads();

	if (i < ny && j < nx) {
		*(u + i*nx + j) = tile_u[ty+1][tx+1];
		*(v + i*nx + j) = tile_v[ty+1][tx+1];
	}



}


int main(){

	char* result_file_name = (char *)"flow_results_cuda.txt";

	int n, it;

	double *ucpu, *vcpu, *pcpu, *bcpu, *pncpu;
	double dx = xmax / (nx-1);
	double dy = ymax / (ny-1);


	struct timeval time_start;
    struct timeval time_end;

	ucpu = (double *) malloc((nx * ny) * sizeof(double));
	vcpu = (double *) malloc((nx * ny) * sizeof(double));
	pcpu = (double *) malloc((nx * ny) * sizeof(double));
	pncpu = (double *) malloc((nx * ny) * sizeof(double));
	bcpu = (double *) malloc((nx * ny) * sizeof(double));

	init(ucpu, vcpu, pcpu, pncpu, bcpu);

	gettimeofday(&time_start, NULL);	


  	double *ugpu, *vgpu, *pgpu, *bgpu, *pngpu, *tempgpu;
    
    cudaMalloc((void **)&ugpu, (nx * ny) * sizeof(double));
    cudaMalloc((void **)&vgpu, (nx * ny) * sizeof(double));
    cudaMalloc((void **)&pgpu, (nx * ny) * sizeof(double));
    cudaMalloc((void **)&pngpu, (nx * ny) * sizeof(double));
    cudaMalloc((void **)&tempgpu, (nx * ny) * sizeof(double));
    cudaMalloc((void **)&bgpu, (nx * ny) * sizeof(double));

    cudaMemcpy(ugpu,ucpu, (nx * ny) * sizeof(double),cudaMemcpyHostToDevice); 
    cudaMemcpy(vgpu,vcpu, (nx * ny) * sizeof(double),cudaMemcpyHostToDevice); 
    cudaMemcpy(pgpu,pcpu, (nx * ny) * sizeof(double),cudaMemcpyHostToDevice); 
    cudaMemcpy(pngpu,pncpu, (nx * ny) * sizeof(double),cudaMemcpyHostToDevice); 
    cudaMemcpy(bgpu,bcpu, (nx * ny) * sizeof(double),cudaMemcpyHostToDevice); 

    dim3 dimGrid(GridSizeX, GridSizeY);
    dim3 dimBlock(BlockSizeX, BlockSizeY);


    for(n =0; n < nt; n++){
        build_up_b<<<dimGrid, dimBlock>>>(bgpu, ugpu, vgpu, dx, dy);

	    for (it = 0; it < nit; it++){
	    	tempgpu = pngpu;
	        pngpu = pgpu;
	        pgpu = tempgpu;
	        solve_pressure_poisson<<<dimGrid, dimBlock>>>(pgpu, pngpu, bgpu, dx, dy);

 	  	 }

       velocity_update<<<dimGrid, dimBlock>>>(ugpu, vgpu, pgpu, dx, dy);
	}

    cudaMemcpy(ucpu,ugpu, (nx * ny) * sizeof(double),cudaMemcpyDeviceToHost); 
    cudaMemcpy(vcpu,vgpu, (nx * ny) * sizeof(double),cudaMemcpyDeviceToHost); 
    cudaMemcpy(pcpu,pgpu, (nx * ny) * sizeof(double),cudaMemcpyDeviceToHost); 
    cudaMemcpy(bcpu,bgpu, (nx * ny) * sizeof(double),cudaMemcpyDeviceToHost); 

    cudaFree(ugpu);
    cudaFree(vgpu);
    cudaFree(pgpu);
    cudaFree(bgpu);


   	gettimeofday(&time_end, NULL);

	save_results(ucpu, vcpu, pcpu, result_file_name, dx, dy);

	free(ucpu);
	free(vcpu);
	free(pcpu);
	free(bcpu);


	double exec_time = (double) (time_end.tv_sec - time_start.tv_sec) +
                   (double) (time_end.tv_usec - time_start.tv_usec) / 1000000.0;

    printf("Running time for CUDA code: %lf\n", exec_time);


	return 0;
}