//
//  ometal.h
//  guetzli_ios
//
//  Created by 张聪 on 2017/9/13.
//  Copyright © 2017年 com.tencent. All rights reserved.
//

#ifdef __USE_METAL__
#import "clguetzli.cl.h"





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


metal_mem allocMem(size_t s, const void *init);


ometal_channels* allocMemChannels(size_t s, const void *c0, const void *c1, const void *c2);







@interface ometal : NSObject

@property (nonatomic, strong) id<MTLDevice> device;
@property (nonatomic, strong) id<MTLCommandQueue> commandQueue;
@property (nonatomic, strong) id<MTLLibrary> defaultLibrary;
@property (nonatomic, strong) id<MTLRenderPipelineState> pipelineState;
@property (nonatomic, strong) NSMutableArray *kernel;

+ (ometal *)sharedInstance;


@end

#endif


