//
//  ometal.m
//  guetzli_ios
//
//  Created by 张聪 on 2017/9/13.
//  Copyright © 2017年 com.tencent. All rights reserved.
//
#ifdef __USE_METAL__

#import "ometal.h"

metal_mem allocMem(size_t s, const void *init )
{
    
    metal_mem mem = nil;
    if (init) {
        ometal *m_ometal = [ometal sharedInstance];
        mem = [m_ometal.device newBufferWithBytes:init
                                           length:s
                                          options:MTLResourceOptionCPUCacheModeDefault];
    }
    
    else
    {
        ometal *m_ometal = [ometal sharedInstance];
        mem = [m_ometal.device newBufferWithLength:s
                                           options:MTLResourceOptionCPUCacheModeDefault];
    }
    if (mem) {
        return mem;
    }
    return NULL;
}

ometal_channels* allocMemChannels(size_t s, const void *c0, const void *c1, const void *c2)
{
    const void *c[3] = { c0, c1, c2 };
    
    ometal_channels* img = new ometal_channels();
    for (int i = 0; i < 3; i++)
    {
        img->ch[i] = allocMem(s, c[i]);
    }
    
    return img;
}


@implementation ometal
static id sharedSingleton = nil;
+ (id)allocWithZone:(struct _NSZone *)zone {
    if (!sharedSingleton) {
        static dispatch_once_t onceToken;
        dispatch_once(&onceToken, ^{
            sharedSingleton = [super allocWithZone:zone];
        });
    }
    return sharedSingleton;
}
- (id)init {
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        sharedSingleton = [super init];
        // Create the default Metal device
        self.device = MTLCreateSystemDefaultDevice();
        // Create a long-lived command queue
        self.commandQueue = [self.device newCommandQueue];
        // Get the library that contains the functions compiled into our app bundle
        self.defaultLibrary = [self.device newDefaultLibrary];
        
        self.kernel = [NSMutableArray arrayWithCapacity:KERNEL_COUNT];
        [self.kernel addObject:[self.defaultLibrary newFunctionWithName:@"clConvolutionEx"]];
        [self.kernel addObject:[self.defaultLibrary newFunctionWithName:@"clConvolutionXEx"]];
        [self.kernel addObject:[self.defaultLibrary newFunctionWithName:@"clConvolutionYEx"]];
        [self.kernel addObject:[self.defaultLibrary newFunctionWithName:@"clSquareSampleEx"]];
        [self.kernel addObject:[self.defaultLibrary newFunctionWithName:@"clOpsinDynamicsImageEx"]];
        [self.kernel addObject:[self.defaultLibrary newFunctionWithName:@"clMaskHighIntensityChangeEx"]];
        [self.kernel addObject:[self.defaultLibrary newFunctionWithName:@"clEdgeDetectorMapEx"]];
        [self.kernel addObject:[self.defaultLibrary newFunctionWithName:@"clBlockDiffMapEx"]];
        [self.kernel addObject:[self.defaultLibrary newFunctionWithName:@"clEdgeDetectorLowFreqEx"]];
        [self.kernel addObject:[self.defaultLibrary newFunctionWithName:@"clDiffPrecomputeEx"]];
        [self.kernel addObject:[self.defaultLibrary newFunctionWithName:@"clScaleImageEx"]];
        [self.kernel addObject:[self.defaultLibrary newFunctionWithName:@"clAverage5x5Ex"]];
        [self.kernel addObject:[self.defaultLibrary newFunctionWithName:@"clMinSquareValEx"]];
        [self.kernel addObject:[self.defaultLibrary newFunctionWithName:@"clDoMaskEx"]];
        [self.kernel addObject:[self.defaultLibrary newFunctionWithName:@"clCombineChannelsEx"]];
        [self.kernel addObject:[self.defaultLibrary newFunctionWithName:@"clUpsampleSquareRootEx"]];
        [self.kernel addObject:[self.defaultLibrary newFunctionWithName:@"clRemoveBorderEx"]];
        [self.kernel addObject:[self.defaultLibrary newFunctionWithName:@"clAddBorderEx"]];
        [self.kernel addObject:[self.defaultLibrary newFunctionWithName:@"clComputeBlockZeroingOrderEx"]];
    });
    return sharedSingleton;
}
+ (instancetype)sharedInstance {
    return [[self alloc] init];
}
+ (id)copyWithZone:(struct _NSZone *)zone {
    return sharedSingleton;
}
+ (id)mutableCopyWithZone:(struct _NSZone *)zone {
    return sharedSingleton;
}
@end
#endif


