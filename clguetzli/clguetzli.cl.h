

#ifdef __USE_METAL__


#import <Foundation/Foundation.h>
#import <Metal/MTLDefines.h>
#import <Metal/MTLBlitCommandEncoder.h>
#import <Metal/MTLBuffer.h>
#import <Metal/MTLCommandBuffer.h>
#import <Metal/MTLComputeCommandEncoder.h>
#import <Metal/MTLCommandQueue.h>
#import <Metal/MTLDevice.h>
#import <Metal/MTLDepthStencil.h>
#import <Metal/MTLDrawable.h>
#import <Metal/MTLRenderPass.h>
#import <Metal/MTLComputePipeline.h>
#import <Metal/MTLLibrary.h>
#import <Metal/MTLPixelFormat.h>
#import <Metal/MTLRenderPipeline.h>
#import <Metal/MTLVertexDescriptor.h>
#import <Metal/MTLParallelRenderCommandEncoder.h>
#import <Metal/MTLRenderCommandEncoder.h>
#import <Metal/MTLSampler.h>
#import <Metal/MTLTexture.h>
#import <Metal/MTLHeap.h>
#define metal_mem id<MTLBuffer>
typedef short coeff_t;
typedef union ometal_channels_t
{
    struct
    {
        metal_mem r =NULL;
        metal_mem g=NULL;
        metal_mem b=NULL;
    };
    struct
    {
        metal_mem x;
        metal_mem y;
        metal_mem b_;
    };
    
    struct
    {
        metal_mem ch[3];
    };
    
    ometal_channels_t() { };
    
}ometal_channels;

typedef struct __channel_info_t
{
    int factor;
    int block_width;
    int block_height;
    const coeff_t *coeff;
    const ushort  *pixel;
}channel_info;

#ifndef __SIMD_HEADER__

#define __SIMD_HEADER__
#endif
#include <simd/vector.h>
#include <simd/matrix.h>



#endif

#ifdef __USE_OPENCL__
#ifndef __APPLE__

#ifdef __cplusplus
#ifndef __CUDACC__
#include "CL/cl.h"
#include "cuda.h"
#endif
#endif

#define __USE_DOUBLE_AS_FLOAT__

#ifdef __cplusplus
#ifndef __CUDACC__
#define __kernel
#define __private
#define __global
#define __constant
#define __constant_ex
#define __device__

typedef unsigned char uchar;
typedef unsigned short ushort;
typedef CUdeviceptr cu_mem;

int get_global_id(int dim);
int get_global_size(int dim);
void set_global_id(int dim, int id);
void set_global_size(int dim, int size);

#ifdef __checkcl
typedef union ocl_channels_t
{
    struct
    {
        float * r;
        float * g;
        float * b;
    };
    union
    {
        float *ch[3];
    };
}ocl_channels;

typedef union ocu_channels_t
{
    struct
    {
        float * r;
        float * g;
        float * b;
    };
    union
    {
        float *ch[3];
    };
}ocu_channels;
#else
typedef union ocl_channels_t
{
    struct
    {
        cl_mem r;
        cl_mem g;
        cl_mem b;
    };
    struct
    {
        cl_mem x;
        cl_mem y;
        cl_mem b_;
    };
    union
    {
        cl_mem ch[3];
    };
}ocl_channels;

typedef union ocu_channels_t
{
    struct
    {
        cu_mem r;
        cu_mem g;
        cu_mem b;
    };
    struct
    {
        cu_mem x;
        cu_mem y;
        cu_mem b_;
    };
    union
    {
        cu_mem ch[3];
    };
}ocu_channels;
#endif
#endif /*__CUDACC__*/
#endif /*__cplusplus*/

#ifdef __OPENCL_VERSION__
#define __constant_ex __constant
#define __device__

#endif /*__OPENCL_VERSION__*/

#ifdef __CUDACC__
#define __kernel    extern "C" __global__
#define __private
#define __global
#define __constant  __constant__
#define __constant_ex
typedef unsigned char uchar;
typedef unsigned short ushort;

__device__ int get_global_id(int dim)
{
    switch (dim)
    {
        case 0:  return blockIdx.x * blockDim.x + threadIdx.x;
        case 1:  return blockIdx.y * blockDim.y + threadIdx.y;
        default: return blockIdx.z * blockDim.z + threadIdx.z;
    }
}

__device__ int get_global_size(int dim)
{
    switch(dim)
    {
        case 0: return gridDim.x * blockDim.x;
        case 1: return gridDim.y * blockDim.y;
        default: return gridDim.z * blockDim.z;
    }
}

#endif /*__CUDACC__*/

typedef short coeff_t;

typedef struct __channel_info_t
{
    int factor;
    int block_width;
    int block_height;
    __global const coeff_t *coeff;
    __global const ushort  *pixel;
}channel_info;



#endif /*__APPLE__*/

#endif /*__USE_OPENCL__*/


