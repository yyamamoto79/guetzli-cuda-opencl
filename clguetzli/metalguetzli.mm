//
//  metalguetzli.cpp
//  guetzli_ios
//
//  Created by 张聪 on 2017/9/13.
//  Copyright © 2017年 com.tencent. All rights reserved.
//
#ifdef __USE_METAL__

#import "metalguetzli.h"
#import "ometal.h"
#import "clguetzli_test.h"
#import "clguetzli.cl.h"
#ifdef __USE_DOUBLE_AS_FLOAT__
#define double float
#endif


#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16
#define BLOCK_COUNT_X(size)    ((size + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X)
#define BLOCK_COUNT_Y(size)    ((size + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y)
#define METAL_ERROR(e)  metalError((e), __FUNCTION__, __LINE__ )

void metalError(NSError *errors,const char* szFunc, int line)
{
    if (errors) {
        NSLog(@"METAL_ERROR %s(%d) %d:%d  errorDescription:%@\r\n", szFunc, line,errors);
        
    }
}



void metalEnqueueReadBuffer(id     /* command_queue */,
                            metal_mem m_metal_mem             /* buffer */,
                            bool             /* blocking_read */,
                            size_t              /* offset */,
                            size_t    size          /* size */,
                            void *   m_ptr           /* ptr */,
                            uint             /* num_events_in_wait_list */,
                            void *    /* event_wait_list */,
                            void *          /* event */)
{
    
    NSData* data = [NSData dataWithBytesNoCopy:[m_metal_mem contents ] length: [m_metal_mem length] freeWhenDone:false ];
    float *values = (float *)[data bytes];
    memcpy ( m_ptr, values,[m_metal_mem length] );
    
}

void metalEnqueueCopyBuffer(id    /* command_queue */,
                            metal_mem src             /* src_buffer */,
                            metal_mem *dst             /* dst_buffer */,
                            size_t              /* src_offset */,
                            size_t              /* dst_offset */,
                            size_t              /* size */,
                            uint             /* num_events_in_wait_list */,
                            void *    /* event_wait_list */,
                            void *          /* event */)
{
    NSData* data = [NSData dataWithBytesNoCopy:[src contents ] length: [src length] freeWhenDone:false ];
    
    *dst = [[ometal sharedInstance].device newBufferWithBytes:[data bytes]
                                                       length:[src length]
                                                      options:MTLResourceOptionCPUCacheModeDefault];
}

void metalOpsinDynamicsImage(float *r, float *g, float *b, const size_t xsize, const size_t ysize)
{
    size_t channel_size = xsize * ysize * sizeof(float);
    ometal *m_ometal = [ometal sharedInstance];
    ometal_channels  * rgb = allocMemChannels(channel_size, r, g, b);
    
    metalOpsinDynamicsImageEx(rgb, xsize, ysize);
    
    metalEnqueueReadBuffer(m_ometal.commandQueue, rgb->r, false, 0, channel_size, r, 0, NULL, NULL);
    metalEnqueueReadBuffer(m_ometal.commandQueue, rgb->g, false, 0, channel_size, g, 0, NULL, NULL);
    metalEnqueueReadBuffer(m_ometal.commandQueue, rgb->b, false, 0, channel_size, b, 0, NULL, NULL);
    
}

void metalDiffmapOpsinDynamicsImage(
                                    float* result,
                                    const float* r,  const float* g,  const float* b,
                                    const float* r2, const float* g2, const float* b2,
                                    const size_t xsize, const size_t ysize,
                                    const size_t step)
{
    size_t channel_size = xsize * ysize * sizeof(float);
    
    ometal *m_ometal = [ometal sharedInstance];
    ometal_channels  *xyb0 = allocMemChannels(channel_size, r, g, b);
    ometal_channels  *xyb1 = allocMemChannels(channel_size, r2, g2, b2);
    
    metal_mem mem_result = allocMem(channel_size, result);
    
    metalDiffmapOpsinDynamicsImageEx(mem_result, xyb0, xyb1, xsize, ysize, step);
    
    metalEnqueueReadBuffer(m_ometal.commandQueue, mem_result, false, 0, channel_size, result, 0, NULL, NULL);
    
}

void metalComputeBlockZeroingOrder(
                                   guetzli::CoeffData *output_order_batch,
                                   const channel_info orig_channel[3],
                                   const float *orig_image_batch,
                                   const float *mask_scale,
                                   const int image_width,
                                   const int image_height,
                                   const channel_info mayout_channel[3],
                                   const int factor,
                                   const int comp_mask,
                                   const float BlockErrorLimit)
{
    const int block8_width = (image_width + 8 - 1) / 8;
    const int block8_height = (image_height + 8 - 1) / 8;
    const int blockf_width = (image_width + 8 * factor - 1) / (8 * factor);
    const int blockf_height = (image_height + 8 * factor - 1) / (8 * factor);
    
    using namespace guetzli;
    
    ometal *m_ometal = [ometal sharedInstance];
    
    metal_mem mem_orig_coeff[3];
    metal_mem mem_mayout_coeff[3];
    metal_mem mem_mayout_pixel[3];
    for (int c = 0; c < 3; c++)
    {
        int block_count = orig_channel[c].block_width * orig_channel[c].block_height;
        mem_orig_coeff[c] = allocMem(block_count * sizeof(::coeff_t) * kDCTBlockSize, orig_channel[c].coeff);
        
        
        block_count = mayout_channel[c].block_width * mayout_channel[c].block_height;
        mem_mayout_coeff[c] = allocMem(block_count * sizeof(::coeff_t) * kDCTBlockSize, mayout_channel[c].coeff);
        
        mem_mayout_pixel[c] = allocMem(image_width * image_height * sizeof(uint16_t), mayout_channel[c].pixel);
    }
    metal_mem mem_orig_image = allocMem(sizeof(float) * 3 * kDCTBlockSize * block8_width * block8_height, orig_image_batch);
    metal_mem mem_mask_scale = allocMem(sizeof(float) * 3 * block8_width * block8_height, mask_scale);
    
    int output_order_batch_size = sizeof(CoeffData) * 3 * kDCTBlockSize * blockf_width * blockf_height;
    metal_mem mem_output_order_batch = allocMem(output_order_batch_size, output_order_batch);
    
    id<MTLBuffer> blockf_widthBuffer =[m_ometal.device newBufferWithBytes:&blockf_width
                                                                   length:sizeof(&blockf_width)
                                                                  options:MTLResourceOptionCPUCacheModeDefault];
    id<MTLBuffer> blockf_heightBuffer =[m_ometal.device newBufferWithBytes:&blockf_height
                                                                    length:sizeof(&blockf_height)
                                                                   options:MTLResourceOptionCPUCacheModeDefault];
    id<MTLBuffer> image_widthBuffer =[m_ometal.device newBufferWithBytes:&image_width
                                                                  length:sizeof(&image_width)
                                                                 options:MTLResourceOptionCPUCacheModeDefault];
    id<MTLBuffer> image_heightBuffer =[m_ometal.device newBufferWithBytes:&image_height
                                                                   length:sizeof(&image_height)
                                                                  options:MTLResourceOptionCPUCacheModeDefault];
    id<MTLBuffer> factorBuffer =[m_ometal.device newBufferWithBytes:&factor
                                                             length:sizeof(&factor)
                                                            options:MTLResourceOptionCPUCacheModeDefault];
    id<MTLBuffer> comp_maskBuffer =[m_ometal.device newBufferWithBytes:&comp_mask
                                                                length:sizeof(&comp_mask)
                                                               options:MTLResourceOptionCPUCacheModeDefault];
    
    id<MTLBuffer> BlockErrorLimitBuffer =[m_ometal.device newBufferWithBytes:&BlockErrorLimit
                                                                      length:sizeof(&BlockErrorLimit)
                                                                     options:MTLResourceOptionCPUCacheModeDefault];
    
    id<MTLBuffer> mayout_channel0Buffer =[m_ometal.device newBufferWithBytes:&mayout_channel[0]
                                                                      length:sizeof(mayout_channel[0])
                                                                     options:MTLResourceOptionCPUCacheModeDefault];
    id<MTLBuffer> mayout_channel1Buffer =[m_ometal.device newBufferWithBytes:&mayout_channel[1]
                                                                      length:sizeof(mayout_channel[1])
                                                                     options:MTLResourceOptionCPUCacheModeDefault];
    id<MTLBuffer> mayout_channel2Buffer =[m_ometal.device newBufferWithBytes:&mayout_channel[2]
                                                                      length:sizeof(mayout_channel[2])
                                                                     options:MTLResourceOptionCPUCacheModeDefault];
    
    // Obtain a new command buffer
    id <MTLCommandBuffer> commandBuffer = [m_ometal.commandQueue commandBuffer];
    
    // Create a compute command encoder
    id <MTLComputeCommandEncoder> computeCE = [commandBuffer computeCommandEncoder];
    id <MTLFunction> kernel =  m_ometal.kernel[KERNEL_COMPUTEBLOCKZEROINGORDER];
    NSError *errors;
    id <MTLComputePipelineState> filterState = [m_ometal.device newComputePipelineStateWithFunction:kernel error:&errors];
    
    [computeCE setComputePipelineState:filterState];
    [computeCE setBuffer:mem_orig_coeff[0] offset:0 atIndex:0];
    [computeCE setBuffer:mem_orig_coeff[1] offset:0 atIndex:1];
    [computeCE setBuffer:mem_orig_coeff[2] offset:0 atIndex:2];
    [computeCE setBuffer:mem_orig_image offset:0 atIndex:3];
    [computeCE setBuffer:mem_mask_scale offset:0 atIndex:4];
    [computeCE setBuffer:blockf_widthBuffer offset:0 atIndex:5];
    [computeCE setBuffer:blockf_heightBuffer offset:0 atIndex:6];
    [computeCE setBuffer:image_widthBuffer offset:0 atIndex:7];
    [computeCE setBuffer:image_heightBuffer offset:0 atIndex:8];
    [computeCE setBuffer:mem_mayout_coeff[0] offset:0 atIndex:9];
    [computeCE setBuffer:mem_mayout_coeff[1] offset:0 atIndex:10];
    [computeCE setBuffer:mem_mayout_coeff[2] offset:0 atIndex:11];
    [computeCE setBuffer:mem_mayout_pixel[0] offset:0 atIndex:12];
    [computeCE setBuffer:mem_mayout_pixel[1] offset:0 atIndex:13];
    [computeCE setBuffer:mem_mayout_pixel[2] offset:0 atIndex:14];
    [computeCE setBuffer:mayout_channel0Buffer offset:0 atIndex:15];
    [computeCE setBuffer:mayout_channel1Buffer offset:0 atIndex:16];
    [computeCE setBuffer:mayout_channel2Buffer offset:0 atIndex:17];
    [computeCE setBuffer:factorBuffer offset:0 atIndex:18];
    [computeCE setBuffer:comp_maskBuffer offset:0 atIndex:19];
    [computeCE setBuffer:BlockErrorLimitBuffer offset:0 atIndex:20];
    [computeCE setBuffer:mem_output_order_batch offset:0 atIndex:21];
    
    
    MTLSize threadsPerGroup = {1, 1, 1};
    MTLSize numThreadgroups = {static_cast<NSUInteger>(blockf_width), static_cast<NSUInteger>(blockf_height), 1};
    [computeCE dispatchThreadgroups:numThreadgroups
              threadsPerThreadgroup:threadsPerGroup];
    [computeCE endEncoding];
    
    // Commit the command buffer
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    METAL_ERROR(errors);
    
}

void metalMask(
               float* mask_r,  float* mask_g,    float* mask_b,
               float* maskdc_r, float* maskdc_g, float* maskdc_b,
               const size_t xsize, const size_t ysize,
               const float* r,  const float* g,  const float* b,
               const float* r2, const float* g2, const float* b2)
{
    ometal *m_ometal = [ometal sharedInstance];
    
    size_t channel_size = xsize * ysize * sizeof(float);
    
    ometal_channels  * rgb = allocMemChannels(channel_size, r, g, b);
    ometal_channels  * rgb2 = allocMemChannels(channel_size, r2, g2, b2);
    ometal_channels  * mask = allocMemChannels(channel_size,NULL,NULL,NULL);
    ometal_channels  * mask_dc = allocMemChannels(channel_size,NULL,NULL,NULL);
    
    metalMaskEx(mask, mask_dc, rgb, rgb2, xsize, ysize);
    
    metalEnqueueReadBuffer(m_ometal.commandQueue, mask->r, false, 0, channel_size, mask_r, 0, NULL, NULL);
    metalEnqueueReadBuffer(m_ometal.commandQueue, mask->g, false, 0, channel_size, mask_g, 0, NULL, NULL);
    metalEnqueueReadBuffer(m_ometal.commandQueue, mask->b, false, 0, channel_size, mask_b, 0, NULL, NULL);
    metalEnqueueReadBuffer(m_ometal.commandQueue, mask_dc->r, false, 0, channel_size, maskdc_r, 0, NULL, NULL);
    metalEnqueueReadBuffer(m_ometal.commandQueue, mask_dc->g, false, 0, channel_size, maskdc_g, 0, NULL, NULL);
    metalEnqueueReadBuffer(m_ometal.commandQueue, mask_dc->b, false, 0, channel_size, maskdc_b, 0, NULL, NULL);
}

void metalDiffmapOpsinDynamicsImageEx(
                                      metal_mem result,
                                      ometal_channels  * xyb0,
                                      ometal_channels  * xyb1,
                                      const size_t xsize, const size_t ysize,
                                      const size_t step)
{
    
    
    const size_t res_xsize = (xsize + step - 1) / step;
    const size_t res_ysize = (ysize + step - 1) / step;
    
    size_t channel_size = xsize * ysize * sizeof(float);
    size_t channel_step_size = res_xsize * res_ysize * sizeof(float);
    
    metal_mem edge_detector_map = allocMem(3 * channel_step_size,NULL);
    metal_mem block_diff_dc = allocMem(3 * channel_step_size,NULL);
    metal_mem block_diff_ac = allocMem(3 * channel_step_size,NULL);
    
    metalMaskHighIntensityChangeEx(xyb0, xyb1, xsize, ysize);
    
    metalEdgeDetectorMapEx(edge_detector_map, xyb0, xyb1, xsize, ysize, step);
    metalBlockDiffMapEx(block_diff_dc, block_diff_ac, xyb0, xyb1, xsize, ysize, step);
    metalEdgeDetectorLowFreqEx(block_diff_ac, xyb0, xyb1, xsize, ysize, step);
    {
        ometal_channels  * mask = allocMemChannels(channel_size,NULL,NULL,NULL);
        ometal_channels  * mask_dc = allocMemChannels(channel_size,NULL,NULL,NULL);
        metalMaskEx(mask, mask_dc, xyb0, xyb1, xsize, ysize);
        metalCombineChannelsEx(result, mask, mask_dc, xsize, ysize, block_diff_dc, block_diff_ac, edge_detector_map, res_xsize, step);
        
    }
    
    
    metalCalculateDiffmapEx(&result, xsize, ysize, step);
    
}


void metalConvolutionEx(
                        metal_mem result/*out*/,
                        const metal_mem inp, size_t xsize, size_t ysize,
                        const metal_mem multipliers, size_t len,
                        int xstep, int offset, float border_ratio)
{
    ometal *m_ometal = [ometal sharedInstance];
    
    size_t oxsize = (xsize + xstep - 1) / xstep;
    
    id <MTLFunction> kernel =  m_ometal.kernel[KERNEL_CONVOLUTION];
    
    id<MTLBuffer> xsizeBuffer =[m_ometal.device newBufferWithBytes:&xsize
                                                            length:sizeof(&xsize)
                                                           options:MTLResourceOptionCPUCacheModeDefault];
    id<MTLBuffer> lenBuffer =[m_ometal.device newBufferWithBytes:&len
                                                          length:sizeof(&len)
                                                         options:MTLResourceOptionCPUCacheModeDefault];
    id<MTLBuffer> xstepBuffer =[m_ometal.device newBufferWithBytes:&xstep
                                                            length:sizeof(&xstep)
                                                           options:MTLResourceOptionCPUCacheModeDefault];
    id<MTLBuffer> offsetBuffer =[m_ometal.device newBufferWithBytes:&offset
                                                             length:sizeof(&offset)
                                                            options:MTLResourceOptionCPUCacheModeDefault];
    id<MTLBuffer> border_ratioBuffer =[m_ometal.device newBufferWithBytes:&border_ratio
                                                                   length:sizeof(&border_ratio)
                                                                  options:MTLResourceOptionCPUCacheModeDefault];
    // Obtain a new command buffer
    id <MTLCommandBuffer> commandBuffer = [m_ometal.commandQueue commandBuffer];
    
    // Create a compute command encoder
    id <MTLComputeCommandEncoder> computeCE = [commandBuffer computeCommandEncoder];
    //        id <MTLFunction> kernel = [m_ometal.defaultLibrary newFunctionWithName:@"metalConvolutionXEx"];
    NSError *errors;
    id <MTLComputePipelineState> filterState = [m_ometal.device newComputePipelineStateWithFunction:kernel error:&errors];
    
    [computeCE setComputePipelineState:filterState];
    [computeCE setBuffer:result offset:0 atIndex:0];
    [computeCE setBuffer:inp offset:0 atIndex:1];
    [computeCE setBuffer:xsizeBuffer offset:0 atIndex:2];
    [computeCE setBuffer:multipliers offset:0 atIndex:3];
    [computeCE setBuffer:lenBuffer offset:0 atIndex:4];
    [computeCE setBuffer:xstepBuffer offset:0 atIndex:5];
    [computeCE setBuffer:offsetBuffer offset:0 atIndex:6];
    [computeCE setBuffer:border_ratioBuffer offset:0 atIndex:7];
    
    
    MTLSize threadsPerGroup = {1, 1, 1};
    MTLSize numThreadgroups= {oxsize, ysize, 1};
    [computeCE dispatchThreadgroups:numThreadgroups
              threadsPerThreadgroup:threadsPerGroup];
    [computeCE endEncoding];
    
    // Commit the command buffer
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    METAL_ERROR(errors);
}

void metalConvolutionXEx(
                         metal_mem result/*out*/,
                         const metal_mem inp, size_t xsize, size_t ysize,
                         const metal_mem multipliers, size_t len,
                         int xstep, int offset, float border_ratio)
{
    ometal *m_ometal = [ometal sharedInstance];
    
    
    
    
    
    
    id<MTLBuffer> xsizeBuffer =[m_ometal.device newBufferWithBytes:&xsize
                                                            length:sizeof(&xsize)
                                                           options:MTLResourceOptionCPUCacheModeDefault];
    id<MTLBuffer> ysizeBuffer =[m_ometal.device newBufferWithBytes:&ysize
                                                            length:sizeof(&ysize)
                                                           options:MTLResourceOptionCPUCacheModeDefault];
    id<MTLBuffer> lenBuffer =[m_ometal.device newBufferWithBytes:&len
                                                          length:sizeof(&len)
                                                         options:MTLResourceOptionCPUCacheModeDefault];
    id<MTLBuffer> xstepBuffer =[m_ometal.device newBufferWithBytes:&xstep
                                                            length:sizeof(&xstep)
                                                           options:MTLResourceOptionCPUCacheModeDefault];
    id<MTLBuffer> offsetBuffer =[m_ometal.device newBufferWithBytes:&offset
                                                             length:sizeof(&offset)
                                                            options:MTLResourceOptionCPUCacheModeDefault];
    id<MTLBuffer> border_ratioBuffer =[m_ometal.device newBufferWithBytes:&border_ratio
                                                                   length:sizeof(&border_ratio)
                                                                  options:MTLResourceOptionCPUCacheModeDefault];
    // Obtain a new command buffer
    id <MTLCommandBuffer> commandBuffer = [m_ometal.commandQueue commandBuffer];
    
    // Create a compute command encoder
    id <MTLComputeCommandEncoder> computeCE = [commandBuffer computeCommandEncoder];
    id <MTLFunction> kernel =  m_ometal.kernel[KERNEL_CONVOLUTIONX];
    //        id <MTLFunction> kernel = [m_ometal.defaultLibrary newFunctionWithName:@"metalConvolutionXEx"];
    NSError *errors;
    id <MTLComputePipelineState> filterState = [m_ometal.device newComputePipelineStateWithFunction:kernel error:&errors];
    
    [computeCE setComputePipelineState:filterState];
    [computeCE setBuffer:result offset:0 atIndex:0];
    [computeCE setBuffer:xsizeBuffer offset:0 atIndex:1];
    [computeCE setBuffer:ysizeBuffer offset:0 atIndex:2];
    [computeCE setBuffer:inp offset:0 atIndex:3];
    [computeCE setBuffer:multipliers offset:0 atIndex:4];
    [computeCE setBuffer:lenBuffer offset:0 atIndex:5];
    [computeCE setBuffer:xstepBuffer offset:0 atIndex:6];
    [computeCE setBuffer:offsetBuffer offset:0 atIndex:7];
    [computeCE setBuffer:border_ratioBuffer offset:0 atIndex:8];
    
    
    MTLSize threadsPerGroup = {1, 1, 1};
    MTLSize numThreadgroups= {xsize, ysize, 1};
    [computeCE dispatchThreadgroups:numThreadgroups
              threadsPerThreadgroup:threadsPerGroup];
    [computeCE endEncoding];
    
    // Commit the command buffer
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    METAL_ERROR(errors);
    
}

void metalConvolutionYEx(
                         metal_mem result/*out*/,
                         const metal_mem inp, size_t xsize, size_t ysize,
                         const metal_mem multipliers, size_t len,
                         int xstep, int offset, float border_ratio)
{
    
    ometal *m_ometal = [ometal sharedInstance];
    
    
    id<MTLBuffer> xsizeBuffer =[m_ometal.device newBufferWithBytes:&xsize
                                                            length:sizeof(&xsize)
                                                           options:MTLResourceOptionCPUCacheModeDefault];
    id<MTLBuffer> ysizeBuffer =[m_ometal.device newBufferWithBytes:&ysize
                                                            length:sizeof(&ysize)
                                                           options:MTLResourceOptionCPUCacheModeDefault];
    id<MTLBuffer> lenBuffer =[m_ometal.device newBufferWithBytes:&len
                                                          length:sizeof(&len)
                                                         options:MTLResourceOptionCPUCacheModeDefault];
    id<MTLBuffer> xstepBuffer =[m_ometal.device newBufferWithBytes:&xstep
                                                            length:sizeof(&xstep)
                                                           options:MTLResourceOptionCPUCacheModeDefault];
    id<MTLBuffer> offsetBuffer =[m_ometal.device newBufferWithBytes:&offset
                                                             length:sizeof(&offset)
                                                            options:MTLResourceOptionCPUCacheModeDefault];
    id<MTLBuffer> border_ratioBuffer =[m_ometal.device newBufferWithBytes:&border_ratio
                                                                   length:sizeof(&border_ratio)
                                                                  options:MTLResourceOptionCPUCacheModeDefault];
    // Obtain a new command buffer
    id <MTLCommandBuffer> commandBuffer = [m_ometal.commandQueue commandBuffer];
    
    // Create a compute command encoder
    id <MTLComputeCommandEncoder> computeCE = [commandBuffer computeCommandEncoder];
    id <MTLFunction> kernel =  m_ometal.kernel[KERNEL_CONVOLUTIONY];
    NSError *errors;
    id <MTLComputePipelineState> filterState
    = [m_ometal.device newComputePipelineStateWithFunction:kernel error:&errors];
    
    [computeCE setComputePipelineState:filterState];
    [computeCE setBuffer:result offset:0 atIndex:0];
    [computeCE setBuffer:xsizeBuffer offset:0 atIndex:1];
    [computeCE setBuffer:ysizeBuffer offset:0 atIndex:2];
    [computeCE setBuffer:inp offset:0 atIndex:3];
    [computeCE setBuffer:multipliers offset:0 atIndex:4];
    [computeCE setBuffer:lenBuffer offset:0 atIndex:5];
    [computeCE setBuffer:xstepBuffer offset:0 atIndex:6];
    [computeCE setBuffer:offsetBuffer offset:0 atIndex:7];
    [computeCE setBuffer:border_ratioBuffer offset:0 atIndex:8];
    
    
    MTLSize threadsPerGroup = {1, 1, 1};
    MTLSize numThreadgroups = {xsize, ysize, 1};
    [computeCE dispatchThreadgroups:numThreadgroups
              threadsPerThreadgroup:threadsPerGroup];
    [computeCE endEncoding];
    
    // Commit the command buffer
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    METAL_ERROR(errors);
}

void metalSquareSampleEx(
                         metal_mem result/*out*/,
                         const metal_mem image, size_t xsize, size_t ysize,
                         size_t xstep, size_t ystep)
{
    ometal *m_ometal = [ometal sharedInstance];
    
    id <MTLFunction> kernel =  m_ometal.kernel[KERNEL_SQUARESAMPLE];
    
    id<MTLBuffer> xsizeBuffer =[m_ometal.device newBufferWithBytes:&xsize
                                                            length:sizeof(&xsize)
                                                           options:MTLResourceOptionCPUCacheModeDefault];
    id<MTLBuffer> ysizeBuffer =[m_ometal.device newBufferWithBytes:&ysize
                                                            length:sizeof(&ysize)
                                                           options:MTLResourceOptionCPUCacheModeDefault];
    id<MTLBuffer> xstepBuffer =[m_ometal.device newBufferWithBytes:&xstep
                                                            length:sizeof(&xstep)
                                                           options:MTLResourceOptionCPUCacheModeDefault];
    id<MTLBuffer> ystepBuffer =[m_ometal.device newBufferWithBytes:&ystep
                                                            length:sizeof(&ystep)
                                                           options:MTLResourceOptionCPUCacheModeDefault];
    // Obtain a new command buffer
    id <MTLCommandBuffer> commandBuffer = [m_ometal.commandQueue commandBuffer];
    
    // Create a compute command encoder
    id <MTLComputeCommandEncoder> computeCE = [commandBuffer computeCommandEncoder];
    
    NSError *errors;
    id <MTLComputePipelineState> filterState
    = [m_ometal.device newComputePipelineStateWithFunction:kernel error:&errors];
    
    [computeCE setComputePipelineState:filterState];
    [computeCE setBuffer:result offset:0 atIndex:0];
    [computeCE setBuffer:xsizeBuffer offset:0 atIndex:1];
    [computeCE setBuffer:ysizeBuffer offset:0 atIndex:2];
    [computeCE setBuffer:image offset:0 atIndex:3];
    [computeCE setBuffer:xstepBuffer offset:0 atIndex:4];
    [computeCE setBuffer:ystepBuffer offset:0 atIndex:5];
    
    
    MTLSize threadsPerGroup = {1, 1, 1};
    MTLSize numThreadgroups = {xsize, ysize, 1};
    [computeCE dispatchThreadgroups:numThreadgroups
              threadsPerThreadgroup:threadsPerGroup];
    [computeCE endEncoding];
    
    // Commit the command buffer
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    METAL_ERROR(errors);
    
}

void metalBlurEx(metal_mem image/*out, opt*/, const size_t xsize, const size_t ysize,
                 const double sigma, const double border_ratio,
                 metal_mem result/*out, opt*/)
{
    double m = 2.25;  // Accuracy increases when m is increased.
    const double scaler = -1.0 / (2 * sigma * sigma);
    // For m = 9.0: exp(-scaler * diff * diff) < 2^ {-52}
    const int diff = std::max<int>(1, m * fabs(sigma));
    const int expn_size = 2 * diff + 1;
    std::vector<float> expn(expn_size);
    for (int i = -diff; i <= diff; ++i) {
        expn[i + diff] = static_cast<float>(exp(scaler * i * i));
    }
    
    const int xstep = std::max<int>(1, int(sigma / 3));
    
    metal_mem mem_expn = allocMem(sizeof(float) * expn_size, expn.data());
    
    if (xstep > 1)
    {
        metal_mem m = allocMem(sizeof(float) * xsize * ysize,NULL);
        metalConvolutionXEx(m, image, xsize, ysize, mem_expn, expn_size, xstep, diff, border_ratio);
        metalConvolutionYEx(result ? result : image, m, xsize, ysize, mem_expn, expn_size, xstep, diff, border_ratio);
        metalSquareSampleEx(result ? result : image, result ? result : image, xsize, ysize, xstep, xstep);
    }
    else
    {
        metal_mem m = allocMem(sizeof(float) * xsize * ysize,NULL);
        metalConvolutionXEx(m, image, xsize, ysize, mem_expn, expn_size, xstep, diff, border_ratio);
        metalConvolutionYEx(result ? result : image, m, xsize, ysize, mem_expn, expn_size, xstep, diff, border_ratio);
    }
    
}

void metalOpsinDynamicsImageEx(ometal_channels  * rgb, const size_t xsize, const size_t ysize)
{
    static const double kSigma = 1.1;
    
    size_t channel_size = xsize * ysize * sizeof(float);
    
    ometal *m_ometal = [ometal sharedInstance];
    ometal_channels  * rgb_blurred = allocMemChannels(channel_size,NULL,NULL,NULL);
    
    const int size = xsize * ysize;
    
    metalBlurEx(rgb->r, xsize, ysize, kSigma, 0.0, rgb_blurred->r);
    metalBlurEx(rgb->g, xsize, ysize, kSigma, 0.0, rgb_blurred->g);
    metalBlurEx(rgb->b, xsize, ysize, kSigma, 0.0, rgb_blurred->b);
    
    id<MTLBuffer> sizeBuffer =[m_ometal.device newBufferWithBytes:&size
                                                           length:sizeof(&size)
                                                          options:MTLResourceOptionCPUCacheModeDefault];
    
    // Obtain a new command buffer
    id <MTLCommandBuffer> commandBuffer = [m_ometal.commandQueue commandBuffer];
    
    // Create a compute command encoder
    id <MTLComputeCommandEncoder> computeCE = [commandBuffer computeCommandEncoder];
    id <MTLFunction> kernel =  m_ometal.kernel[KERNEL_OPSINDYNAMICSIMAGE];    NSError *errors;
    id <MTLComputePipelineState> filterState
    = [m_ometal.device newComputePipelineStateWithFunction:kernel error:&errors];
    
    [computeCE setComputePipelineState:filterState];
    [computeCE setBuffer:rgb->r offset:0 atIndex:0];
    [computeCE setBuffer:rgb->g offset:0 atIndex:1];
    [computeCE setBuffer:rgb->b offset:0 atIndex:2];
    [computeCE setBuffer:sizeBuffer offset:0 atIndex:3];
    [computeCE setBuffer:rgb_blurred->r offset:0 atIndex:4];
    [computeCE setBuffer:rgb_blurred->g offset:0 atIndex:5];
    [computeCE setBuffer:rgb_blurred->b offset:0 atIndex:6];
    
    MTLSize threadsPerGroup = {1,1, 1};
    MTLSize numThreadgroups = {xsize*ysize, 1, 1};
    [computeCE dispatchThreadgroups:numThreadgroups
              threadsPerThreadgroup:threadsPerGroup];
    [computeCE endEncoding];
    
    // Commit the command buffer
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    METAL_ERROR(errors);
    
}

void metalMaskHighIntensityChangeEx(
                                    ometal_channels  * xyb0/*in,out*/,
                                    ometal_channels  * xyb1/*in,out*/,
                                    const size_t xsize, const size_t ysize)
{
    size_t channel_size = xsize * ysize * sizeof(float);
    
    ometal *m_ometal = [ometal sharedInstance];
    
    ometal_channels  * c0 = allocMemChannels(channel_size,NULL,NULL,NULL);
    ometal_channels  * c1 = allocMemChannels(channel_size,NULL,NULL,NULL);
    metal_mem  c0r= c0->r;
    metal_mem  c0g= c0->g;
    metal_mem  c0b= c0->b;
    metal_mem  c1r= c1->r;
    metal_mem  c1g= c1->g;
    metal_mem  c1b= c1->b;
    metalEnqueueCopyBuffer(m_ometal.commandQueue, xyb0->r, &c0r, 0, 0, channel_size, 0, NULL, NULL);
    //NSData* data = [NSData dataWithBytesNoCopy:[c0r contents ] length: [c0r length] freeWhenDone:false ];
    
    metalEnqueueCopyBuffer(m_ometal.commandQueue, xyb0->g, &c0g, 0, 0, channel_size, 0, NULL, NULL);
    metalEnqueueCopyBuffer(m_ometal.commandQueue, xyb0->b, &c0b, 0, 0, channel_size, 0, NULL, NULL);
    metalEnqueueCopyBuffer(m_ometal.commandQueue, xyb1->r, &c1r, 0, 0, channel_size, 0, NULL, NULL);
    metalEnqueueCopyBuffer(m_ometal.commandQueue, xyb1->g, &c1g, 0, 0, channel_size, 0, NULL, NULL);
    metalEnqueueCopyBuffer(m_ometal.commandQueue, xyb1->b, &c1b, 0, 0, channel_size, 0, NULL, NULL);
    
    
    
    
    id <MTLFunction> kernel =  m_ometal.kernel[KERNEL_MASKHIGHINTENSITYCHANGE];
    
    
    
    id<MTLBuffer> xsizeBuffer =[m_ometal.device newBufferWithBytes:&xsize
                                                            length:sizeof(&xsize)
                                                           options:MTLResourceOptionCPUCacheModeDefault];
    id<MTLBuffer> ysizeBuffer =[m_ometal.device newBufferWithBytes:&ysize
                                                            length:sizeof(&ysize)
                                                           options:MTLResourceOptionCPUCacheModeDefault];
    
    
    // Obtain a new command buffer
    id <MTLCommandBuffer> commandBuffer = [m_ometal.commandQueue commandBuffer];
    
    // Create a compute command encoder
    id <MTLComputeCommandEncoder> computeCE = [commandBuffer computeCommandEncoder];
    
    NSError *errors;
    id <MTLComputePipelineState> filterState
    = [m_ometal.device newComputePipelineStateWithFunction:kernel error:&errors];
    
    [computeCE setComputePipelineState:filterState];
    [computeCE setBuffer:xyb0->r offset:0 atIndex:0];
    [computeCE setBuffer:xyb0->g offset:0 atIndex:1];
    [computeCE setBuffer:xyb0->b offset:0 atIndex:2];
    [computeCE setBuffer:xsizeBuffer offset:0 atIndex:3];
    [computeCE setBuffer:ysizeBuffer offset:0 atIndex:4];
    [computeCE setBuffer:xyb1->r offset:0 atIndex:5];
    [computeCE setBuffer:xyb1->g offset:0 atIndex:6];
    [computeCE setBuffer:xyb1->b offset:0 atIndex:7];
    [computeCE setBuffer:c0r offset:0 atIndex:8];
    [computeCE setBuffer:c0g offset:0 atIndex:9];
    [computeCE setBuffer:c0b offset:0 atIndex:10];
    [computeCE setBuffer:c1r offset:0 atIndex:11];
    [computeCE setBuffer:c1g offset:0 atIndex:12];
    [computeCE setBuffer:c1b offset:0 atIndex:13];
    
    MTLSize threadsPerGroup = {1,1, 1};
    MTLSize numThreadgroups= {xsize, ysize, 1};
    [computeCE dispatchThreadgroups:numThreadgroups
              threadsPerThreadgroup:threadsPerGroup];
    [computeCE endEncoding];
    
    // Commit the command buffer
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    METAL_ERROR(errors);
    
}

void metalEdgeDetectorMapEx(
                            metal_mem result/*out*/,
                            const ometal_channels  * rgb, const ometal_channels  * rgb2,
                            const size_t xsize, const size_t ysize, const size_t step)
{
    
    size_t channel_size = xsize * ysize * sizeof(float);
    
    ometal *m_ometal = [ometal sharedInstance];
    ometal_channels  * rgb_blured = allocMemChannels(channel_size,NULL,NULL,NULL);
    ometal_channels  * rgb2_blured = allocMemChannels(channel_size,NULL,NULL,NULL);
    
    static const double kSigma[3] = { 1.5, 0.586, 0.4 };
    
    for (int i = 0; i < 3; i++)
    {
        metalBlurEx(rgb->ch[i], xsize, ysize, kSigma[i], 0.0, rgb_blured->ch[i]);
        metalBlurEx(rgb2->ch[i], xsize, ysize, kSigma[i], 0.0, rgb2_blured->ch[i]);
    }
    
    const size_t res_xsize = (xsize + step - 1) / step;
    const size_t res_ysize = (ysize + step - 1) / step;
    
    
    id<MTLBuffer> res_xsizeBuffer =[m_ometal.device newBufferWithBytes:&res_xsize
                                                                length:sizeof(&res_xsize)
                                                               options:MTLResourceOptionCPUCacheModeDefault];
    id<MTLBuffer> res_ysizeBuffer =[m_ometal.device newBufferWithBytes:&res_ysize
                                                                length:sizeof(&res_ysize)
                                                               options:MTLResourceOptionCPUCacheModeDefault];
    id<MTLBuffer> xsizeBuffer =[m_ometal.device newBufferWithBytes:&xsize
                                                            length:sizeof(&xsize)
                                                           options:MTLResourceOptionCPUCacheModeDefault];
    id<MTLBuffer> ysizeBuffer =[m_ometal.device newBufferWithBytes:&ysize
                                                            length:sizeof(&ysize)
                                                           options:MTLResourceOptionCPUCacheModeDefault];
    id<MTLBuffer> stepBuffer =[m_ometal.device newBufferWithBytes:&step
                                                           length:sizeof(&step)
                                                          options:MTLResourceOptionCPUCacheModeDefault];
    
    // Obtain a new command buffer
    id <MTLCommandBuffer> commandBuffer = [m_ometal.commandQueue commandBuffer];
    
    // Create a compute command encoder
    id <MTLComputeCommandEncoder> computeCE = [commandBuffer computeCommandEncoder];
    id <MTLFunction> kernel =  m_ometal.kernel[KERNEL_EDGEDETECTOR];
    NSError *errors;
    id <MTLComputePipelineState> filterState
    = [m_ometal.device newComputePipelineStateWithFunction:kernel error:&errors];
    
    [computeCE setComputePipelineState:filterState];
    [computeCE setBuffer:result offset:0 atIndex:0];
    [computeCE setBuffer:res_xsizeBuffer offset:0 atIndex:1];
    [computeCE setBuffer:res_ysizeBuffer offset:0 atIndex:2];
    [computeCE setBuffer:rgb_blured->r offset:0 atIndex:3];
    [computeCE setBuffer:rgb_blured->g offset:0 atIndex:4];
    [computeCE setBuffer:rgb_blured->b offset:0 atIndex:5];
    [computeCE setBuffer:rgb2_blured->r offset:0 atIndex:6];
    [computeCE setBuffer:rgb2_blured->g offset:0 atIndex:7];
    [computeCE setBuffer:rgb2_blured->b offset:0 atIndex:8];
    [computeCE setBuffer:xsizeBuffer offset:0 atIndex:9];
    [computeCE setBuffer:ysizeBuffer offset:0 atIndex:10];
    [computeCE setBuffer:stepBuffer offset:0 atIndex:11];
    
    MTLSize threadsPerGroup = {1,1, 1};
    MTLSize numThreadgroups = {res_xsize, res_ysize, 1};
    [computeCE dispatchThreadgroups:numThreadgroups
              threadsPerThreadgroup:threadsPerGroup];
    [computeCE endEncoding];
    
    // Commit the command buffer
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    METAL_ERROR(errors);
    
}

void metalBlockDiffMapEx(
                         metal_mem block_diff_dc/*out*/,
                         metal_mem block_diff_ac/*out*/,
                         const ometal_channels  *rgb, const ometal_channels  *rgb2,
                         const size_t xsize, const size_t ysize, const size_t step)
{
    ometal *m_ometal = [ometal sharedInstance];
    
    
    const size_t res_xsize = (xsize + step - 1) / step;
    const size_t res_ysize = (ysize + step - 1) / step;
    
    id <MTLFunction> kernel =  m_ometal.kernel[KERNEL_BLOCKDIFFMAP];
    
    id<MTLBuffer> res_xsizeBuffer =[m_ometal.device newBufferWithBytes:&res_xsize
                                                                length:sizeof(&res_xsize)
                                                               options:MTLResourceOptionCPUCacheModeDefault];
    id<MTLBuffer> res_ysizeBuffer =[m_ometal.device newBufferWithBytes:&res_ysize
                                                                length:sizeof(&res_ysize)
                                                               options:MTLResourceOptionCPUCacheModeDefault];
    id<MTLBuffer> xsizeBuffer =[m_ometal.device newBufferWithBytes:&xsize
                                                            length:sizeof(&xsize)
                                                           options:MTLResourceOptionCPUCacheModeDefault];
    id<MTLBuffer> ysizeBuffer =[m_ometal.device newBufferWithBytes:&ysize
                                                            length:sizeof(&ysize)
                                                           options:MTLResourceOptionCPUCacheModeDefault];
    id<MTLBuffer> stepBuffer =[m_ometal.device newBufferWithBytes:&step
                                                           length:sizeof(&step)
                                                          options:MTLResourceOptionCPUCacheModeDefault];
    
    // Obtain a new command buffer
    id <MTLCommandBuffer> commandBuffer = [m_ometal.commandQueue commandBuffer];
    
    // Create a compute command encoder
    id <MTLComputeCommandEncoder> computeCE = [commandBuffer computeCommandEncoder];
    NSError *errors;
    id <MTLComputePipelineState> filterState
    = [m_ometal.device newComputePipelineStateWithFunction:kernel error:&errors];
    [computeCE setComputePipelineState:filterState];
    
    [computeCE setBuffer:block_diff_dc offset:0 atIndex:0];
    [computeCE setBuffer:block_diff_ac offset:0 atIndex:1];
    [computeCE setBuffer:res_xsizeBuffer offset:0 atIndex:2];
    [computeCE setBuffer:res_ysizeBuffer offset:0 atIndex:3];
    [computeCE setBuffer:rgb->r offset:0 atIndex:4];
    [computeCE setBuffer:rgb->g offset:0 atIndex:5];
    [computeCE setBuffer:rgb->b offset:0 atIndex:6];
    [computeCE setBuffer:rgb2->r offset:0 atIndex:7];
    [computeCE setBuffer:rgb2->g offset:0 atIndex:8];
    [computeCE setBuffer:rgb2->b offset:0 atIndex:9];
    [computeCE setBuffer:xsizeBuffer offset:0 atIndex:10];
    [computeCE setBuffer:ysizeBuffer offset:0 atIndex:11];
    [computeCE setBuffer:stepBuffer offset:0 atIndex:12];
    
    MTLSize threadsPerGroup = {1,1, 1};
    MTLSize numThreadgroups = {res_xsize, res_ysize, 1};
    [computeCE dispatchThreadgroups:numThreadgroups
              threadsPerThreadgroup:threadsPerGroup];
    [computeCE endEncoding];
    
    // Commit the command buffer
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    METAL_ERROR(errors);
    
}


void metalEdgeDetectorLowFreqEx(
                                metal_mem block_diff_ac/*in,out*/,
                                const ometal_channels  *rgb, const ometal_channels  *rgb2,
                                const size_t xsize, const size_t ysize, const size_t step)
{
    size_t channel_size = xsize * ysize * sizeof(float);
    
    static const double kSigma = 14;
    ometal *m_ometal = [ometal sharedInstance];
    ometal_channels  * rgb_blured = allocMemChannels(channel_size,NULL,NULL,NULL);
    ometal_channels  * rgb2_blured = allocMemChannels(channel_size,NULL,NULL,NULL);
    
    for (int i = 0; i < 3; i++)
    {
        metalBlurEx(rgb->ch[i], xsize, ysize, kSigma, 0.0, rgb_blured->ch[i]);
        metalBlurEx(rgb2->ch[i], xsize, ysize, kSigma, 0.0, rgb2_blured->ch[i]);
    }
    
    const size_t res_xsize = (xsize + step - 1) / step;
    const size_t res_ysize = (ysize + step - 1) / step;
    
    id <MTLFunction> kernel =  m_ometal.kernel[KERNEL_EDGEDETECTORLOWFREQ];
    
    id<MTLBuffer> res_xsizeBuffer =[m_ometal.device newBufferWithBytes:&res_xsize
                                                                length:sizeof(&res_xsize)
                                                               options:MTLResourceOptionCPUCacheModeDefault];
    id<MTLBuffer> res_ysizeBuffer =[m_ometal.device newBufferWithBytes:&res_ysize
                                                                length:sizeof(&res_ysize)
                                                               options:MTLResourceOptionCPUCacheModeDefault];
    id<MTLBuffer> xsizeBuffer =[m_ometal.device newBufferWithBytes:&xsize
                                                            length:sizeof(&xsize)
                                                           options:MTLResourceOptionCPUCacheModeDefault];
    id<MTLBuffer> ysizeBuffer =[m_ometal.device newBufferWithBytes:&ysize
                                                            length:sizeof(&ysize)
                                                           options:MTLResourceOptionCPUCacheModeDefault];
    id<MTLBuffer> stepBuffer =[m_ometal.device newBufferWithBytes:&step
                                                           length:sizeof(&step)
                                                          options:MTLResourceOptionCPUCacheModeDefault];
    
    // Obtain a new command buffer
    id <MTLCommandBuffer> commandBuffer = [m_ometal.commandQueue commandBuffer];
    
    // Create a compute command encoder
    id <MTLComputeCommandEncoder> computeCE = [commandBuffer computeCommandEncoder];
    NSError *errors;
    id <MTLComputePipelineState> filterState
    = [m_ometal.device newComputePipelineStateWithFunction:kernel error:&errors];
    
    [computeCE setComputePipelineState:filterState];
    [computeCE setBuffer:block_diff_ac offset:0 atIndex:0];
    [computeCE setBuffer:res_xsizeBuffer offset:0 atIndex:1];
    [computeCE setBuffer:res_ysizeBuffer offset:0 atIndex:2];
    [computeCE setBuffer:rgb_blured->r offset:0 atIndex:3];
    [computeCE setBuffer:rgb_blured->g offset:0 atIndex:4];
    [computeCE setBuffer:rgb_blured->b offset:0 atIndex:5];
    [computeCE setBuffer:rgb2_blured->r offset:0 atIndex:6];
    [computeCE setBuffer:rgb2_blured->g offset:0 atIndex:7];
    [computeCE setBuffer:rgb2_blured->b offset:0 atIndex:8];
    [computeCE setBuffer:xsizeBuffer offset:0 atIndex:9];
    [computeCE setBuffer:ysizeBuffer offset:0 atIndex:10];
    [computeCE setBuffer:stepBuffer offset:0 atIndex:11];
    
    MTLSize threadsPerGroup = {1,1, 1};
    MTLSize numThreadgroups = {res_xsize, res_ysize, 1};
    [computeCE dispatchThreadgroups:numThreadgroups
              threadsPerThreadgroup:threadsPerGroup];
    [computeCE endEncoding];
    
    // Commit the command buffer
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    METAL_ERROR(errors);
    
}

void metalDiffPrecomputeEx(
                           ometal_channels  *mask/*out*/,
                           const ometal_channels  *xyb0, const ometal_channels  *xyb1,
                           const size_t xsize, const size_t ysize)
{
    
    ometal *m_ometal = [ometal sharedInstance];
    
    
    id <MTLFunction> kernel =  m_ometal.kernel[KERNEL_DIFFPRECOMPUTE];
    
    id<MTLBuffer> xsizeBuffer =[m_ometal.device newBufferWithBytes:&xsize
                                                            length:sizeof(&xsize)
                                                           options:MTLResourceOptionCPUCacheModeDefault];
    id<MTLBuffer> ysizeBuffer =[m_ometal.device newBufferWithBytes:&ysize
                                                            length:sizeof(&ysize)
                                                           options:MTLResourceOptionCPUCacheModeDefault];
    
    // Obtain a new command buffer
    id <MTLCommandBuffer> commandBuffer = [m_ometal.commandQueue commandBuffer];
    
    // Create a compute command encoder
    id <MTLComputeCommandEncoder> computeCE = [commandBuffer computeCommandEncoder];
    NSError *errors;
    id <MTLComputePipelineState> filterState
    = [m_ometal.device newComputePipelineStateWithFunction:kernel error:&errors];
    
    [computeCE setComputePipelineState:filterState];
    [computeCE setBuffer:mask->x offset:0 atIndex:0];
    [computeCE setBuffer:mask->y offset:0 atIndex:1];
    [computeCE setBuffer:mask->b offset:0 atIndex:2];
    [computeCE setBuffer:xsizeBuffer offset:0 atIndex:3];
    [computeCE setBuffer:ysizeBuffer offset:0 atIndex:4];
    [computeCE setBuffer:xyb0->x offset:0 atIndex:5];
    [computeCE setBuffer:xyb0->y offset:0 atIndex:6];
    [computeCE setBuffer:xyb0->b offset:0 atIndex:7];
    [computeCE setBuffer:xyb1->x offset:0 atIndex:8];
    [computeCE setBuffer:xyb1->y offset:0 atIndex:9];
    [computeCE setBuffer:xyb1->b offset:0 atIndex:10];
    
    
    MTLSize threadsPerGroup = {1,1,1};
    MTLSize numThreadgroups = {xsize, ysize, 1};
    [computeCE dispatchThreadgroups:numThreadgroups
              threadsPerThreadgroup:threadsPerGroup];
    [computeCE endEncoding];
    
    // Commit the command buffer
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    METAL_ERROR(errors);
    
}

void metalScaleImageEx(metal_mem img/*in, out*/, size_t size, double w)
{
    
    ometal *m_ometal = [ometal sharedInstance];
    float fw = w;
    
    id <MTLFunction> kernel =  m_ometal.kernel[KERNEL_SCALEIMAGE];
    
    id<MTLBuffer> sizeBuffer =[m_ometal.device newBufferWithBytes:&size
                                                           length:sizeof(&size)
                                                          options:MTLResourceOptionCPUCacheModeDefault];
    id<MTLBuffer> fwBuffer =[m_ometal.device newBufferWithBytes:&fw
                                                         length:sizeof(&fw)
                                                        options:MTLResourceOptionCPUCacheModeDefault];
    
    // Obtain a new command buffer
    id <MTLCommandBuffer> commandBuffer = [m_ometal.commandQueue commandBuffer];
    
    // Create a compute command encoder
    id <MTLComputeCommandEncoder> computeCE = [commandBuffer computeCommandEncoder];
    NSError *errors;
    id <MTLComputePipelineState> filterState
    = [m_ometal.device newComputePipelineStateWithFunction:kernel error:&errors];
    
    [computeCE setComputePipelineState:filterState];
    [computeCE setBuffer:img offset:0 atIndex:0];
    [computeCE setBuffer:sizeBuffer offset:0 atIndex:1];
    [computeCE setBuffer:fwBuffer offset:0 atIndex:2];
    
    MTLSize threadsPerGroup = {1,1, 1};
    MTLSize numThreadgroups = {size, 1, 1};
    [computeCE dispatchThreadgroups:numThreadgroups
              threadsPerThreadgroup:threadsPerGroup];
    [computeCE endEncoding];
    
    // Commit the command buffer
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    METAL_ERROR(errors);
}

void metalAverage5x5Ex(metal_mem img/*in,out*/, const size_t xsize, const size_t ysize)
{
    if (xsize < 4 || ysize < 4) {
        // TODO: Make this work for small dimensions as well.
        return;
    }
    
    ometal *m_ometal = [ometal sharedInstance];
    
    size_t len = xsize * ysize * sizeof(float);
    metal_mem img_org = allocMem(len,NULL);
    
    metalEnqueueCopyBuffer(m_ometal.commandQueue, img, &img_org, 0, 0, len, 0, NULL, NULL);
    
    id <MTLFunction> kernel =  m_ometal.kernel[KERNEL_AVERAGE5X5];
    id <MTLCommandBuffer> commandBuffer = [m_ometal.commandQueue commandBuffer];
    
    // Create a compute command encoder
    id <MTLComputeCommandEncoder> computeCE = [commandBuffer computeCommandEncoder];
    id<MTLBuffer> xsizeBuffer =[m_ometal.device newBufferWithBytes:&xsize
                                                            length:sizeof(&xsize)
                                                           options:MTLResourceOptionCPUCacheModeDefault];
    id<MTLBuffer> ysizeBuffer =[m_ometal.device newBufferWithBytes:&ysize
                                                            length:sizeof(&ysize)
                                                           options:MTLResourceOptionCPUCacheModeDefault];
    NSError *errors;
    id <MTLComputePipelineState> filterState
    = [m_ometal.device newComputePipelineStateWithFunction:kernel error:&errors];
    
    
    [computeCE setComputePipelineState:filterState];
    [computeCE setBuffer:img offset:0 atIndex:0];
    [computeCE setBuffer:xsizeBuffer offset:0 atIndex:1];
    [computeCE setBuffer:ysizeBuffer offset:0 atIndex:2];
    [computeCE setBuffer:img_org offset:0 atIndex:3];
    
    MTLSize threadsPerGroup = {1,1, 1};
    MTLSize numThreadgroups = {xsize, ysize, 1};
    [computeCE dispatchThreadgroups:numThreadgroups
              threadsPerThreadgroup:threadsPerGroup];
    [computeCE endEncoding];
    
    // Commit the command buffer
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    METAL_ERROR(errors);
    
}

metal_mem metalMinSquareValEx(
                              metal_mem img/*in,out*/,
                              const size_t xsize, const size_t ysize,
                              const size_t square_size, const size_t offset,int unuse)
{
    ometal *m_ometal = [ometal sharedInstance];
    
    metal_mem result = allocMem(sizeof(float) * xsize * ysize,NULL);
    
    id <MTLFunction> kernel =  m_ometal.kernel[KERNEL_MINSQUAREVAL];
    id <MTLCommandBuffer> commandBuffer = [m_ometal.commandQueue commandBuffer];
    
    // Create a compute command encoder
    id <MTLComputeCommandEncoder> computeCE = [commandBuffer computeCommandEncoder];
    id<MTLBuffer> xsizeBuffer =[m_ometal.device newBufferWithBytes:&xsize
                                                            length:sizeof(&xsize)
                                                           options:MTLResourceOptionCPUCacheModeDefault];
    id<MTLBuffer> ysizeBuffer =[m_ometal.device newBufferWithBytes:&ysize
                                                            length:sizeof(&ysize)
                                                           options:MTLResourceOptionCPUCacheModeDefault];
    id<MTLBuffer> square_sizeBuffer =[m_ometal.device newBufferWithBytes:&square_size
                                                                  length:sizeof(&square_size)
                                                                 options:MTLResourceOptionCPUCacheModeDefault];
    id<MTLBuffer> offsetBuffer =[m_ometal.device newBufferWithBytes:&offset
                                                             length:sizeof(&offset)
                                                            options:MTLResourceOptionCPUCacheModeDefault];
    NSError *errors;
    id <MTLComputePipelineState> filterState
    = [m_ometal.device newComputePipelineStateWithFunction:kernel error:&errors];
    
    
    [computeCE setComputePipelineState:filterState];
    [computeCE setBuffer:result offset:0 atIndex:0];
    [computeCE setBuffer:xsizeBuffer offset:0 atIndex:1];
    [computeCE setBuffer:ysizeBuffer offset:0 atIndex:2];
    [computeCE setBuffer:img offset:0 atIndex:3];
    [computeCE setBuffer:square_sizeBuffer offset:0 atIndex:4];
    [computeCE setBuffer:offsetBuffer offset:0 atIndex:5];
    
    MTLSize threadsPerGroup = {1,1, 1};
    MTLSize numThreadgroups = {xsize, ysize, 1};
    [computeCE dispatchThreadgroups:numThreadgroups
              threadsPerThreadgroup:threadsPerGroup];
    [computeCE endEncoding];
    
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    METAL_ERROR(errors);
    
    return result;
    
}

static void MakeMask(double extmul, double extoff,
                     double mul, double offset,
                     double scaler, double *result)
{
    for (size_t i = 0; i < 512; ++i) {
        const double c = mul / ((0.01 * scaler * i) + offset);
        result[i] = 1.0 + extmul * (c + extoff);
        result[i] *= result[i];
    }
}

static const double kInternalGoodQualityThreshold = 14.921561160295326;
static const double kGlobalScale = 1.0 / kInternalGoodQualityThreshold;

void metalDoMask(ometal_channels  * mask/*in, out*/, ometal_channels  * mask_dc/*in, out*/, size_t xsize, size_t ysize)
{
    ometal *m_ometal = [ometal sharedInstance];
    
    double extmul = 0.975741017749;
    double extoff = -4.25328244168;
    double offset = 0.454909521427;
    double scaler = 0.0738288224836;
    double mul = 20.8029176447;
    static double lut_x[512];
    static bool lutx_init = false;
    if (!lutx_init)
    {
        lutx_init = true;
        MakeMask(extmul, extoff, mul, offset, scaler, lut_x);
    }
    
    extmul = 0.373995618954;
    extoff = 1.5307267433;
    offset = 0.911952641929;
    scaler = 1.1731667845;
    mul = 16.2447033988;
    static double lut_y[512];
    static bool luty_init = false;
    if (!luty_init)
    {
        luty_init = true;
        MakeMask(extmul, extoff, mul, offset, scaler, lut_y);
    }
    
    extmul = 0.61582234137;
    extoff = -4.25376118646;
    offset = 1.05105070921;
    scaler = 0.47434643535;
    mul = 31.1444967089;
    static double lut_b[512];
    static bool lutb_init = false;
    if (!lutb_init)
    {
        lutb_init = true;
        MakeMask(extmul, extoff, mul, offset, scaler, lut_b);
    }
    
    extmul = 1.79116943438;
    extoff = -3.86797479189;
    offset = 0.670960225853;
    scaler = 0.486575865525;
    mul = 20.4563479139;
    static double lut_dcx[512];
    static bool lutdcx_init = false;
    if (!lutdcx_init)
    {
        lutdcx_init = true;
        MakeMask(extmul, extoff, mul, offset, scaler, lut_dcx);
    }
    
    extmul = 0.212223514236;
    extoff = -3.65647120524;
    offset = 1.73396799447;
    scaler = 0.170392660501;
    mul = 21.6566724788;
    static double lut_dcy[512];
    static bool lutdcy_init = false;
    if (!lutdcy_init)
    {
        lutdcy_init = true;
        MakeMask(extmul, extoff, mul, offset, scaler, lut_dcy);
    }
    
    extmul = 0.349376011816;
    extoff = -0.894711072781;
    offset = 0.901647926679;
    scaler = 0.380086095024;
    mul = 18.0373825149;
    static double lut_dcb[512];
    static bool lutdcb_init = false;
    if (!lutdcb_init)
    {
        lutdcb_init = true;
        MakeMask(extmul, extoff, mul, offset, scaler, lut_dcb);
    }
    
    static float lut_xf[512] = {0};
    static float lut_yf[512] = {0};
    static float lut_bf[512] = {0};
    static float lut_dcxf[512] = {0};
    static float lut_dcyf[512] = {0};
    static float lut_dcbf[512] = {0};
    for (int i =0 ; i<512; i++) {
        lut_xf[i] = lut_x[i] ;
        lut_yf[i] = lut_y[i] ;
        lut_bf[i] = lut_b[i] ;
        lut_dcxf[i] = lut_dcx[i] ;
        lut_dcyf[i] = lut_dcy[i] ;
        lut_dcbf[i] = lut_dcb[i] ;
    }
    
    size_t channel_size = 512 * sizeof(float);
    ometal_channels  * xyb = allocMemChannels(channel_size, lut_xf, lut_yf, lut_bf);
    ometal_channels  * xyb_dc = allocMemChannels(channel_size, lut_dcxf, lut_dcyf, lut_dcbf);
    
    id <MTLFunction> kernel =  m_ometal.kernel[KERNEL_DOMASK];
    id <MTLCommandBuffer> commandBuffer = [m_ometal.commandQueue commandBuffer];
    
    // Create a compute command encoder
    id <MTLComputeCommandEncoder> computeCE = [commandBuffer computeCommandEncoder];
    id<MTLBuffer> xsizeBuffer =[m_ometal.device newBufferWithBytes:&xsize
                                                            length:sizeof(&xsize)
                                                           options:MTLResourceOptionCPUCacheModeDefault];
    id<MTLBuffer> ysizeBuffer =[m_ometal.device newBufferWithBytes:&ysize
                                                            length:sizeof(&ysize)
                                                           options:MTLResourceOptionCPUCacheModeDefault];
    NSError *errors;
    id <MTLComputePipelineState> filterState
    = [m_ometal.device newComputePipelineStateWithFunction:kernel error:&errors];
    
    
    [computeCE setComputePipelineState:filterState];
    [computeCE setBuffer:mask->r offset:0 atIndex:0];
    [computeCE setBuffer:mask->g offset:0 atIndex:1];
    [computeCE setBuffer:mask->b offset:0 atIndex:2];
    [computeCE setBuffer:xsizeBuffer offset:0 atIndex:3];
    [computeCE setBuffer:ysizeBuffer offset:0 atIndex:4];
    [computeCE setBuffer:mask_dc->r offset:0 atIndex:5];
    [computeCE setBuffer:mask_dc->g offset:0 atIndex:6];
    [computeCE setBuffer:mask_dc->b offset:0 atIndex:7];
    [computeCE setBuffer:xyb->r offset:0 atIndex:8];
    [computeCE setBuffer:xyb->g offset:0 atIndex:9];
    [computeCE setBuffer:xyb->b offset:0 atIndex:10];
    [computeCE setBuffer:xyb_dc->r offset:0 atIndex:11];
    [computeCE setBuffer:xyb_dc->g offset:0 atIndex:12];
    [computeCE setBuffer:xyb_dc->b offset:0 atIndex:13];
    
    
    MTLSize threadsPerGroup = {1,1, 1};
    MTLSize numThreadgroups = {xsize, ysize, 1};
    [computeCE dispatchThreadgroups:numThreadgroups
              threadsPerThreadgroup:threadsPerGroup];
    [computeCE endEncoding];
    
    // Commit the command buffer
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    METAL_ERROR(errors);
    
    
}

void metalMaskEx(
                 ometal_channels  * mask/*out*/, ometal_channels  * mask_dc/*out*/,
                 const ometal_channels  *rgb, const ometal_channels  *rgb2,
                 const size_t xsize, const size_t ysize)
{
    
    metalDiffPrecomputeEx(mask, rgb, rgb2, xsize, ysize);
    for (int i = 0; i < 3; i++)
    {
        metalAverage5x5Ex(mask->ch[i], xsize, ysize);
        mask->ch[i] = metalMinSquareValEx(mask->ch[i], xsize, ysize, 4, 0,0);
        
        static const double sigma[3] = {
            9.65781083553,
            14.2644604355,
            4.53358927369,
        };
        {
            
            metalBlurEx(mask->ch[i], xsize, ysize, sigma[i], 0.0);
        }
        
    }
    //到此为止数据正常
    
    
    metalDoMask(mask, mask_dc, xsize, ysize);//出了问题
    
    //从这里开始数据不对
    for (int i = 0; i < 3; i++)
    {
        
        {
            metalScaleImageEx(mask->ch[i], xsize * ysize, kGlobalScale * kGlobalScale);
            metalScaleImageEx(mask_dc->ch[i], xsize * ysize, kGlobalScale * kGlobalScale);
        }
        
        
        
    }
}

void metalCombineChannelsEx(
                            metal_mem result/*out*/,
                            const ometal_channels  *mask,
                            const ometal_channels  *mask_dc,
                            const size_t xsize, const size_t ysize,
                            const metal_mem block_diff_dc,
                            const metal_mem block_diff_ac,
                            const metal_mem edge_detector_map,
                            const size_t res_xsize,
                            const size_t step)
{
    ometal *m_ometal = [ometal sharedInstance];
    
    const size_t work_xsize = ((xsize - 8 + step) + step - 1) / step;
    const size_t work_ysize = ((ysize - 8 + step) + step - 1) / step;
    
    id <MTLFunction> kernel =  m_ometal.kernel[KERNEL_COMBINECHANNELS];
    
    
    id <MTLCommandBuffer> commandBuffer = [m_ometal.commandQueue commandBuffer];
    
    // Create a compute command encoder
    id <MTLComputeCommandEncoder> computeCE = [commandBuffer computeCommandEncoder];
    id<MTLBuffer> xsizeBuffer =[m_ometal.device newBufferWithBytes:&xsize
                                                            length:sizeof(&xsize)
                                                           options:MTLResourceOptionCPUCacheModeDefault];
    id<MTLBuffer> ysizeBuffer =[m_ometal.device newBufferWithBytes:&ysize
                                                            length:sizeof(&ysize)
                                                           options:MTLResourceOptionCPUCacheModeDefault];
    id<MTLBuffer> res_xsizeBuffer =[m_ometal.device newBufferWithBytes:&res_xsize
                                                                length:sizeof(&res_xsize)
                                                               options:MTLResourceOptionCPUCacheModeDefault];
    id<MTLBuffer> stepBuffer =[m_ometal.device newBufferWithBytes:&step
                                                           length:sizeof(&step)
                                                          options:MTLResourceOptionCPUCacheModeDefault];
    NSError *errors;
    id <MTLComputePipelineState> filterState
    = [m_ometal.device newComputePipelineStateWithFunction:kernel error:&errors];
    
    
    [computeCE setComputePipelineState:filterState];
    [computeCE setBuffer:result offset:0 atIndex:0];
    [computeCE setBuffer:mask->r offset:0 atIndex:1];
    [computeCE setBuffer:mask->g offset:0 atIndex:2];
    [computeCE setBuffer:mask->b offset:0 atIndex:3];
    [computeCE setBuffer:mask_dc->r offset:0 atIndex:4];
    [computeCE setBuffer:mask_dc->g offset:0 atIndex:5];
    [computeCE setBuffer:mask_dc->b offset:0 atIndex:6];
    [computeCE setBuffer:xsizeBuffer offset:0 atIndex:7];
    [computeCE setBuffer:ysizeBuffer offset:0 atIndex:8];
    [computeCE setBuffer:block_diff_dc offset:0 atIndex:9];
    [computeCE setBuffer:block_diff_ac offset:0 atIndex:10];
    [computeCE setBuffer:edge_detector_map offset:0 atIndex:11];
    [computeCE setBuffer:res_xsizeBuffer offset:0 atIndex:12];
    [computeCE setBuffer:stepBuffer offset:0 atIndex:13];
    
    
    MTLSize threadsPerGroup = {1,1, 1};
    MTLSize numThreadgroups = {work_xsize, work_ysize, 1};
    [computeCE dispatchThreadgroups:numThreadgroups
              threadsPerThreadgroup:threadsPerGroup];
    [computeCE endEncoding];
    
    // Commit the command buffer
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    METAL_ERROR(errors);
    
}

void metalUpsampleSquareRootEx(metal_mem *diffmap, const size_t xsize, const size_t ysize, const int step)
{
    
    
    
    ometal *m_ometal = [ometal sharedInstance];
    
    metal_mem diffmap_out = allocMem(xsize * ysize * sizeof(float),NULL);
    
    id <MTLFunction> kernel =  m_ometal.kernel[KERNEL_UPSAMPLESQUAREROOT];
    
    const size_t res_xsize = (xsize + step - 1) / step;
    const size_t res_ysize = (ysize + step - 1) / step;
    
    id<MTLBuffer> xsizeBuffer =[m_ometal.device newBufferWithBytes:&xsize
                                                            length:sizeof(&xsize)
                                                           options:MTLResourceOptionCPUCacheModeDefault];
    id<MTLBuffer> ysizeBuffer =[m_ometal.device newBufferWithBytes:&ysize
                                                            length:sizeof(&ysize)
                                                           options:MTLResourceOptionCPUCacheModeDefault];
    id<MTLBuffer> stepBuffer =[m_ometal.device newBufferWithBytes:&step
                                                           length:sizeof(&step)
                                                          options:MTLResourceOptionCPUCacheModeDefault];
    
    // Obtain a new command buffer
    id <MTLCommandBuffer> commandBuffer = [m_ometal.commandQueue commandBuffer];
    
    // Create a compute command encoder
    id <MTLComputeCommandEncoder> computeCE = [commandBuffer computeCommandEncoder];
    NSError *errors;
    id <MTLComputePipelineState> filterState
    = [m_ometal.device newComputePipelineStateWithFunction:kernel error:&errors];
    
    [computeCE setComputePipelineState:filterState];
    [computeCE setBuffer:diffmap_out offset:0 atIndex:0];
    [computeCE setBuffer:*diffmap offset:0 atIndex:1];
    [computeCE setBuffer:xsizeBuffer offset:0 atIndex:2];
    [computeCE setBuffer:ysizeBuffer offset:0 atIndex:3];
    [computeCE setBuffer:stepBuffer offset:0 atIndex:4];
    
    
    
    MTLSize threadsPerGroup = {1,1, 1};
    MTLSize numThreadgroups = {res_xsize, res_ysize, 1};
    [computeCE dispatchThreadgroups:numThreadgroups
              threadsPerThreadgroup:threadsPerGroup];
    [computeCE endEncoding];
    
    // Commit the command buffer
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    METAL_ERROR(errors);
    
    metalEnqueueCopyBuffer(m_ometal.commandQueue, diffmap_out, diffmap, 0, 0, xsize * ysize * sizeof(float), 0, NULL, NULL);
}

void metalRemoveBorderEx(metal_mem out, const metal_mem in, const size_t xsize, const size_t ysize, const int step)
{
    ometal *m_ometal = [ometal sharedInstance];
    
    int metals = 8 - step;
    int metals2 = (8 - step) / 2;
    
    size_t out_xsize = xsize - metals;
    size_t out_ysize = ysize - metals;
    
    id <MTLFunction> kernel =  m_ometal.kernel[KERNEL_REMOVEBORDER];
    
    
    id <MTLCommandBuffer> commandBuffer = [m_ometal.commandQueue commandBuffer];
    
    // Create a compute command encoder
    id <MTLComputeCommandEncoder> computeCE = [commandBuffer computeCommandEncoder];
    id<MTLBuffer> out_xsizeBuffer =[m_ometal.device newBufferWithBytes:&out_xsize
                                                                length:sizeof(&out_xsize)
                                                               options:MTLResourceOptionCPUCacheModeDefault];
    id<MTLBuffer> out_ysizeBuffer =[m_ometal.device newBufferWithBytes:&out_ysize
                                                                length:sizeof(&out_ysize)
                                                               options:MTLResourceOptionCPUCacheModeDefault];
    id<MTLBuffer> metalsBuffer =[m_ometal.device newBufferWithBytes:&metals
                                                             length:sizeof(&metals)
                                                            options:MTLResourceOptionCPUCacheModeDefault];
    id<MTLBuffer> metals2Buffer =[m_ometal.device newBufferWithBytes:&metals2
                                                              length:sizeof(&metals2)
                                                             options:MTLResourceOptionCPUCacheModeDefault];
    NSError *errors;
    id <MTLComputePipelineState> filterState
    = [m_ometal.device newComputePipelineStateWithFunction:kernel error:&errors];
    
    
    [computeCE setComputePipelineState:filterState];
    [computeCE setBuffer:out offset:0 atIndex:0];
    [computeCE setBuffer:out_xsizeBuffer offset:0 atIndex:1];
    [computeCE setBuffer:out_ysizeBuffer offset:0 atIndex:2];
    [computeCE setBuffer:in offset:0 atIndex:3];
    [computeCE setBuffer:metalsBuffer offset:0 atIndex:4];
    [computeCE setBuffer:metals2Buffer offset:0 atIndex:5];
    
    
    
    MTLSize threadsPerGroup = {1,1, 1};
    MTLSize numThreadgroups = {out_xsize, out_ysize, 1};
    [computeCE dispatchThreadgroups:numThreadgroups
              threadsPerThreadgroup:threadsPerGroup];
    [computeCE endEncoding];
    
    // Commit the command buffer
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    METAL_ERROR(errors);
    
}

void metalAddBorderEx(metal_mem out, size_t xsize, size_t ysize, int step, metal_mem in)
{
    ometal *m_ometal = [ometal sharedInstance];
    
    int metals = 8 - step;
    int metals2 = (8 - step) / 2;
    id <MTLFunction> kernel =  m_ometal.kernel[KERNEL_ADDBORDER];
    id <MTLCommandBuffer> commandBuffer = [m_ometal.commandQueue commandBuffer];
    
    // Create a compute command encoder
    id <MTLComputeCommandEncoder> computeCE = [commandBuffer computeCommandEncoder];
    id<MTLBuffer> xsizeBuffer =[m_ometal.device newBufferWithBytes:&xsize
                                                            length:sizeof(&xsize)
                                                           options:MTLResourceOptionCPUCacheModeDefault];
    id<MTLBuffer> ysizeBuffer =[m_ometal.device newBufferWithBytes:&ysize
                                                            length:sizeof(&ysize)
                                                           options:MTLResourceOptionCPUCacheModeDefault];
    id<MTLBuffer> metalsBuffer =[m_ometal.device newBufferWithBytes:&metals
                                                             length:sizeof(&metals)
                                                            options:MTLResourceOptionCPUCacheModeDefault];
    id<MTLBuffer> metals2Buffer =[m_ometal.device newBufferWithBytes:&metals2
                                                              length:sizeof(&metals2)
                                                             options:MTLResourceOptionCPUCacheModeDefault];
    NSError *errors;
    id <MTLComputePipelineState> filterState
    = [m_ometal.device newComputePipelineStateWithFunction:kernel error:&errors];
    
    
    [computeCE setComputePipelineState:filterState];
    [computeCE setBuffer:out offset:0 atIndex:0];
    [computeCE setBuffer:xsizeBuffer offset:0 atIndex:1];
    [computeCE setBuffer:ysizeBuffer offset:0 atIndex:2];
    [computeCE setBuffer:metalsBuffer offset:0 atIndex:3];
    [computeCE setBuffer:metals2Buffer offset:0 atIndex:4];
    [computeCE setBuffer:in offset:0 atIndex:5];
    
    
    
    MTLSize threadsPerGroup = {1,1, 1};
    MTLSize numThreadgroups = {xsize, ysize, 1};
    [computeCE dispatchThreadgroups:numThreadgroups
              threadsPerThreadgroup:threadsPerGroup];
    [computeCE endEncoding];
    
    // Commit the command buffer
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    METAL_ERROR(errors);
}

void metalCalculateDiffmapEx(metal_mem *diffmap/*in,out*/, const size_t xsize, const size_t ysize, const int step)
{
    
    
    metalUpsampleSquareRootEx(diffmap, xsize, ysize, step);
    static const double kSigma = 8.8510880283;
    static const double mul1 = 24.8235314874;
    static const double scale = 1.0 / (1.0 + mul1);
    
    const int s = 8 - step;
    int s2 = (8 - step) / 2;
    
    metal_mem blurred = allocMem((xsize - s) * (ysize - s) * sizeof(float),NULL);
    metalRemoveBorderEx(blurred, *diffmap, xsize, ysize, step);
    
    static const double border_ratio = 0.03027655136;
    metalBlurEx(blurred, xsize - s, ysize - s, kSigma, border_ratio);
    
    metalAddBorderEx(*diffmap, xsize, ysize, step, blurred);
    metalScaleImageEx(*diffmap, xsize * ysize, scale);
    
}
#ifdef __USE_DOUBLE_AS_FLOAT__
#undef double
#endif

#endif

