//
//  ometal.h
//  guetzli_ios_metal
//
//  Created by 张聪 on 2017/8/15.
//  Copyright © 2017年 张聪. All rights reserved.
//

//#include "ocl.h"

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


#ifndef __SIMD_HEADER__
#define __SIMD_HEADER__

#include <simd/vector.h>
#include <simd/matrix.h>

#endif

@class metalguezli;

#define cl_mem id<MTLBuffer>

typedef union ocl_channels_t
{
    struct
    {
         cl_mem r =NULL;
         cl_mem g=NULL;
         cl_mem b=NULL;
    };
    struct
    {
        cl_mem x;
        cl_mem y;
        cl_mem b_;
    };

    struct
    {
        cl_mem ch[3];
    };
    
    ocl_channels_t() { };

}ocl_channels;

typedef short coeff_t;


typedef struct __channel_info_t
{
    int factor;
    int block_width;
    int block_height;
    const coeff_t *coeff;
    const ushort  *pixel;
}channel_info;

enum KernelName {
    KERNEL_CONVOLUTION = 0,
    KERNEL_CONVOLUTIONX,
    KERNEL_CONVOLUTIONY,
    KERNEL_SQUARESAMPLE,
    KERNEL_OPSINDYNAMICSIMAGE,
    KERNEL_MASKHIGHINTENSITYCHANGE,
    KERNEL_EDGEDETECTOR,
    KERNEL_BLOCKDIFFMAP,
    KERNEL_EDGEDETECTORLOWFREQ,
    KERNEL_DIFFPRECOMPUTE,
    KERNEL_SCALEIMAGE,
    KERNEL_AVERAGE5X5,
    KERNEL_MINSQUAREVAL,
    KERNEL_DOMASK,
    KERNEL_COMBINECHANNELS,
    KERNEL_UPSAMPLESQUAREROOT,
    KERNEL_REMOVEBORDER,
    KERNEL_ADDBORDER,
    KERNEL_COMPUTEBLOCKZEROINGORDER,
    KERNEL_COUNT,
};

#define LOG_CL_RESULT(e)   if (CL_SUCCESS != (e)) { LogError("Error: %s:%d returned %s.\n", __FUNCTION__, __LINE__, TranslateOpenCLError((e)));}


cl_mem allocMem(size_t s, const void *init);


ocl_channels* allocMemChannels(size_t s, const void *c0, const void *c1, const void *c2);


void releaseMemChannels(ocl_channels &rgb);






@interface ometal : NSObject

@property (nonatomic, strong) id<MTLDevice> device;
@property (nonatomic, strong) id<MTLCommandQueue> commandQueue;
@property (nonatomic, strong) id<MTLLibrary> defaultLibrary;
@property (nonatomic, strong) id<MTLRenderPipelineState> pipelineState;
@property (nonatomic, strong) NSMutableArray *kernel;

+ (ometal *)sharedInstance;


@end
