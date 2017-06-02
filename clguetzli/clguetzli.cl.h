#ifndef __CLGUETZLI_CL_H__
#define __CLGUETZLI_CL_H__

#ifdef __cplusplus
#ifndef __CUDACC__
#include "CL\cl.h"
#include "cuda.h"
#endif
#endif

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
                CUdeviceptr r;
                CUdeviceptr g;
                CUdeviceptr b;
            };
            struct
            {
                CUdeviceptr x;
                CUdeviceptr y;
                CUdeviceptr b_;
            };
            union
            {
                CUdeviceptr ch[3];
            };
        }ocu_channels;
    #endif
#endif /*__CUDACC__*/
#endif /*__cplusplus*/

#ifdef __OPENCL_VERSION__
    #define __constant_ex __constant
    #define __device__
/*
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
*/
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
        case 0:  return blockIdx.x;
        case 1:  return blockIdx.y;
        default: return blockIdx.z;
        }
    }

    __device__ int get_global_size(int dim)
    {
        switch(dim)
        {
        case 0: return gridDim.x;
        case 1: return gridDim.y;
        default: return gridDim.z;
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

#endif /*__CLGUETZLI_CL_H__*/