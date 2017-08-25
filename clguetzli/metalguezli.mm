//
//  metalguezli.m
//  guetzli_ios_metal
//
//  Created by 张聪 on 2017/8/15.
//  Copyright © 2017年 张聪. All rights reserved.
//

#import "metalguezli.h"
#import "ometal.h"
#ifdef __USE_DOUBLE_AS_FLOAT__
#define double float
#endif

#define cuFinish cuStreamSynchronize
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16
#define BLOCK_COUNT_X(size)    ((size + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X)
#define BLOCK_COUNT_Y(size)    ((size + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y)


void clEnqueueReadBuffer(id     /* command_queue */,
                         cl_mem m_cl_mem             /* buffer */,
                         bool             /* blocking_read */,
                         size_t              /* offset */,
                         size_t    size          /* size */,
                         void *   m_ptr           /* ptr */,
                         uint             /* num_events_in_wait_list */,
                         void *    /* event_wait_list */,
                         void *          /* event */)
{
    
    NSData* data = [NSData dataWithBytesNoCopy:[m_cl_mem contents ] length: [m_cl_mem length] freeWhenDone:false ];
    void *values = (void *)[data bytes];
    memcpy ( m_ptr, values, sizeof([m_cl_mem length]) );
}

void clEnqueueCopyBuffer(id    /* command_queue */,
                         cl_mem src             /* src_buffer */,
                         cl_mem dst             /* dst_buffer */,
                         size_t              /* src_offset */,
                         size_t              /* dst_offset */,
                         size_t              /* size */,
                         uint             /* num_events_in_wait_list */,
                         void *    /* event_wait_list */,
                         void *          /* event */)
{
    NSData* data = [NSData dataWithBytesNoCopy:[src contents ] length: [src length] freeWhenDone:false ];
    
    dst = [[ometal sharedInstance].device newBufferWithBytes:[data bytes]
                                                      length:[src length]
                                                     options:MTLResourceOptionCPUCacheModeDefault];
}

void clOpsinDynamicsImage(float *r, float *g, float *b, const size_t xsize, const size_t ysize)
{
    size_t channel_size = xsize * ysize * sizeof(float);
    ometal *m_ometal = [ometal sharedInstance];
    
    ocl_channels  * rgb = allocMemChannels(channel_size, r, g, b);
    
    clOpsinDynamicsImageEx(rgb, xsize, ysize);
    
    clEnqueueReadBuffer(m_ometal.commandQueue, rgb->r, false, 0, channel_size, r, 0, NULL, NULL);
    clEnqueueReadBuffer(m_ometal.commandQueue, rgb->g, false, 0, channel_size, g, 0, NULL, NULL);
    clEnqueueReadBuffer(m_ometal.commandQueue, rgb->b, false, 0, channel_size, b, 0, NULL, NULL);
    //   clFinish(m_ometal.commandQueue);
    
    //    m_ometal.releaseMemChannels(rgb);
}

void clDiffmapOpsinDynamicsImage(
                                 float* result,
                                 const float* r,  const float* g,  const float* b,
                                 const float* r2, const float* g2, const float* b2,
                                 const size_t xsize, const size_t ysize,
                                 const size_t step)
{
    size_t channel_size = xsize * ysize * sizeof(float);
    
    ometal *m_ometal = [ometal sharedInstance];
    ocl_channels  *xyb0 = allocMemChannels(channel_size, r, g, b);
    ocl_channels  *xyb1 = allocMemChannels(channel_size, r2, g2, b2);
    
    cl_mem mem_result = allocMem(channel_size, result);
    
    clDiffmapOpsinDynamicsImageEx(mem_result, xyb0, xyb1, xsize, ysize, step);
    
    clEnqueueReadBuffer(m_ometal.commandQueue, mem_result, false, 0, channel_size, result, 0, NULL, NULL);
    //   int err = clFinish(m_ometal.commandQueue);
    
    //    m_ometal.releaseMemChannels(xyb1);
    //    m_ometal.releaseMemChannels(xyb0);
    
    //    clReleaseMemObject(mem_result);
}

void clComputeBlockZeroingOrder(
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
    
    cl_mem mem_orig_coeff[3];
    cl_mem mem_mayout_coeff[3];
    cl_mem mem_mayout_pixel[3];
    for (int c = 0; c < 3; c++)
    {
        int block_count = orig_channel[c].block_width * orig_channel[c].block_height;
        mem_orig_coeff[c] = allocMem(block_count * sizeof(::coeff_t) * kDCTBlockSize, orig_channel[c].coeff);

        
        block_count = mayout_channel[c].block_width * mayout_channel[c].block_height;
        mem_mayout_coeff[c] = allocMem(block_count * sizeof(::coeff_t) * kDCTBlockSize, mayout_channel[c].coeff);
        
        mem_mayout_pixel[c] = allocMem(image_width * image_height * sizeof(uint16_t), mayout_channel[c].pixel);
    }
    cl_mem mem_orig_image = allocMem(sizeof(float) * 3 * kDCTBlockSize * block8_width * block8_height, orig_image_batch);
    cl_mem mem_mask_scale = allocMem(sizeof(float) * 3 * block8_width * block8_height, mask_scale);
    
    int output_order_batch_size = sizeof(CoeffData) * 3 * kDCTBlockSize * blockf_width * blockf_height;
    cl_mem mem_output_order_batch = allocMem(output_order_batch_size, output_order_batch);

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
    
}

void clMask(
            float* mask_r,  float* mask_g,    float* mask_b,
            float* maskdc_r, float* maskdc_g, float* maskdc_b,
            const size_t xsize, const size_t ysize,
            const float* r,  const float* g,  const float* b,
            const float* r2, const float* g2, const float* b2)
{
    ometal *m_ometal = [ometal sharedInstance];
    
    size_t channel_size = xsize * ysize * sizeof(float);
    
    ocl_channels  * rgb = allocMemChannels(channel_size, r, g, b);
    ocl_channels  * rgb2 = allocMemChannels(channel_size, r2, g2, b2);
    ocl_channels  * mask = allocMemChannels(channel_size,NULL,NULL,NULL);
    ocl_channels  * mask_dc = allocMemChannels(channel_size,NULL,NULL,NULL);
    
    clMaskEx(mask, mask_dc, rgb, rgb2, xsize, ysize);
    
    clEnqueueReadBuffer(m_ometal.commandQueue, mask->r, false, 0, channel_size, mask_r, 0, NULL, NULL);
    clEnqueueReadBuffer(m_ometal.commandQueue, mask->g, false, 0, channel_size, mask_g, 0, NULL, NULL);
    clEnqueueReadBuffer(m_ometal.commandQueue, mask->b, false, 0, channel_size, mask_b, 0, NULL, NULL);
    clEnqueueReadBuffer(m_ometal.commandQueue, mask_dc->r, false, 0, channel_size, maskdc_r, 0, NULL, NULL);
    clEnqueueReadBuffer(m_ometal.commandQueue, mask_dc->g, false, 0, channel_size, maskdc_g, 0, NULL, NULL);
    clEnqueueReadBuffer(m_ometal.commandQueue, mask_dc->b, false, 0, channel_size, maskdc_b, 0, NULL, NULL);
}

void clDiffmapOpsinDynamicsImageEx(
                                   cl_mem result,
                                   ocl_channels  * xyb0,
                                   ocl_channels  * xyb1,
                                   const size_t xsize, const size_t ysize,
                                   const size_t step)
{

    
    const size_t res_xsize = (xsize + step - 1) / step;
    const size_t res_ysize = (ysize + step - 1) / step;
    
    size_t channel_size = xsize * ysize * sizeof(float);
    size_t channel_step_size = res_xsize * res_ysize * sizeof(float);
    
    ometal *m_ometal = [ometal sharedInstance];
    
    cl_mem edge_detector_map = allocMem(3 * channel_step_size,NULL);
    cl_mem block_diff_dc = allocMem(3 * channel_step_size,NULL);
    cl_mem block_diff_ac = allocMem(3 * channel_step_size,NULL);
    
    clMaskHighIntensityChangeEx(xyb0, xyb1, xsize, ysize);
    
    clEdgeDetectorMapEx(edge_detector_map, xyb0, xyb1, xsize, ysize, step);
    clBlockDiffMapEx(block_diff_dc, block_diff_ac, xyb0, xyb1, xsize, ysize, step);
    clEdgeDetectorLowFreqEx(block_diff_ac, xyb0, xyb1, xsize, ysize, step);
    {
        ocl_channels  * mask = allocMemChannels(channel_size,NULL,NULL,NULL);
        ocl_channels  * mask_dc = allocMemChannels(channel_size,NULL,NULL,NULL);
        clMaskEx(mask, mask_dc, xyb0, xyb1, xsize, ysize);
        clCombineChannelsEx(result, mask, mask_dc, xsize, ysize, block_diff_dc, block_diff_ac, edge_detector_map, res_xsize, step);
                
    }
    
    NSData* data1 = [NSData dataWithBytesNoCopy:[edge_detector_map contents ] length: [edge_detector_map length] freeWhenDone:false ];
    NSData* data2 = [NSData dataWithBytesNoCopy:[block_diff_ac contents ] length: [block_diff_ac length] freeWhenDone:false ];
    
    clCalculateDiffmapEx(result, xsize, ysize, step);
    NSData* data = [NSData dataWithBytesNoCopy:[result contents ] length: [result length] freeWhenDone:false ];
}

//#define test

#ifdef test
//#define kernel
#define DEVICE
//#define device
#define THREAD
#define thread
#define constant

double Interpolate(constant const double *array, const int size, const double sx) {
    double ix = fabs(sx);
    
    int baseix = (int)(ix);
    double res;
    if (baseix >= size - 1) {
        res = array[size - 1];
    }
    else {
        double mix = ix - baseix;
        int nextix = baseix + 1;
        res = array[baseix] + mix * (array[nextix] - array[baseix]);
    }
    if (sx < 0) res = -res;
    return res;
}
#define XybLowFreqToVals_inc 5.2511644570349185
constant double XybLowFreqToVals_lut[21] = {
    0,
    1 * XybLowFreqToVals_inc,
    2 * XybLowFreqToVals_inc,
    3 * XybLowFreqToVals_inc,
    4 * XybLowFreqToVals_inc,
    5 * XybLowFreqToVals_inc,
    6 * XybLowFreqToVals_inc,
    7 * XybLowFreqToVals_inc,
    8 * XybLowFreqToVals_inc,
    9 * XybLowFreqToVals_inc,
    10 * XybLowFreqToVals_inc,
    11 * XybLowFreqToVals_inc,
    12 * XybLowFreqToVals_inc,
    13 * XybLowFreqToVals_inc,
    14 * XybLowFreqToVals_inc,
    15 * XybLowFreqToVals_inc,
    16 * XybLowFreqToVals_inc,
    17 * XybLowFreqToVals_inc,
    18 * XybLowFreqToVals_inc,
    19 * XybLowFreqToVals_inc,
    20 * XybLowFreqToVals_inc,
};

void XybLowFreqToVals(double x, double y, double z,
                      thread double *valx, thread double *valy,thread  double *valz) {
    const double xmul = 6.64482198135;
    const double ymul = 0.837846224276;
    const double zmul = 7.34905756986;
    const double y_to_z_mul = 0.0812519812628;
    
    z += y_to_z_mul * y;
    *valz = z * zmul;
    *valx = x * xmul;
    *valy = Interpolate(&XybLowFreqToVals_lut[0], 21, y * ymul);
}

void XybDiffLowFreqSquaredAccumulate(double r0, double g0, double b0,
                                     double r1, double g1, double b1,
                                     double factor, double res[3]) {
    double valx0, valy0, valz0;
    double valx1, valy1, valz1;
    
    XybLowFreqToVals(r0, g0, b0, &valx0, &valy0, &valz0);
    if (r1 == 0.0 && g1 == 0.0 && b1 == 0.0) {
        //PROFILER_ZONE("XybDiff r1=g1=b1=0");
        res[0] += factor * valx0 * valx0;
        res[1] += factor * valy0 * valy0;
        res[2] += factor * valz0 * valz0;
        return;
    }
    XybLowFreqToVals(r1, g1, b1, &valx1, &valy1, &valz1);
    // Approximate the distance of the colors by their respective distances
    // to gray.
    double valx = valx0 - valx1;
    double valy = valy0 - valy1;
    double valz = valz0 - valz1;
    res[0] += factor * valx * valx;
    res[1] += factor * valy * valy;
    res[2] += factor * valz * valz;
}

void Butteraugli8x8CornerEdgeDetectorDiff(
                                          int pos_x,
                                          int pos_y,
                                          int xsize,
                                          int ysize,
                                          DEVICE const float *r, DEVICE const float *g, DEVICE const float* b,
                                          DEVICE const float *r2, DEVICE const float* g2, DEVICE const float *b2,
                                          THREAD double* diff_xyb)
{
    int local_count = 0;
    double local_xyb[3] = { 0 };
    const double w = 0.711100840192;
    
    int offset[4][2] = { { 0,0 },{ 0,7 },{ 7,0 },{ 7,7 } };
    int edgeSize = 3;
    
    for (int k = 0; k < 4; k++)
    {
        int x = pos_x + offset[k][0];
        int y = pos_y + offset[k][1];
        
        if (x >= edgeSize && x + edgeSize < xsize) {
            size_t ix = y * xsize + (x - edgeSize);
            size_t ix2 = ix + 2 * edgeSize;
            XybDiffLowFreqSquaredAccumulate(
                                            w * (r[ix] - r[ix2]),
                                            w * (g[ix] - g[ix2]),
                                            w * (b[ix] - b[ix2]),
                                            w * (r2[ix] - r2[ix2]),
                                            w * (g2[ix] - g2[ix2]),
                                            w * (b2[ix] - b2[ix2]),
                                            1.0, local_xyb);
            ++local_count;
        }
        if (y >= edgeSize && y + edgeSize < ysize) {
            size_t ix = (y - edgeSize) * xsize + x;
            size_t ix2 = ix + 2 * edgeSize * xsize;
            XybDiffLowFreqSquaredAccumulate(
                                            w * (r[ix] - r[ix2]),
                                            w * (g[ix] - g[ix2]),
                                            w * (b[ix] - b[ix2]),
                                            w * (r2[ix] - r2[ix2]),
                                            w * (g2[ix] - g2[ix2]),
                                            w * (b2[ix] - b2[ix2]),
                                            1.0, local_xyb);
            ++local_count;
        }
    }
    
    const double weight = 0.01617112696;
    const double mul = weight * 8.0 / local_count;
    for (int i = 0; i < 3; ++i) {
        diff_xyb[i] += mul * local_xyb[i];
    }
}

 void clEdgeDetectorMapEx(
                                DEVICE float *result,
                                DEVICE  int *res_xsizenum, DEVICE  int *res_ysizenum,
                                DEVICE  float *r, DEVICE  float *g, DEVICE  float* b,
                                DEVICE  float *r2, DEVICE  float* g2, DEVICE  float *b2,
                                DEVICE int *xsizenum, DEVICE int *ysizenum, DEVICE int *stepnum,int x,int y)
{
    
    int step = *stepnum;
    int xsize = *xsizenum;
    int ysize = *ysizenum;
    int res_xsize = *res_xsizenum;
    int res_ysize = *res_ysizenum;
    
    const int res_x = x;
    const int res_y = y;
    
    if (res_x >= res_xsize || res_y >= res_ysize) return;
    
    int pos_x = res_x * step;
    int pos_y = res_y * step;
    
    if (pos_x >= xsize - (8 - step)) return;
    if (pos_y >= ysize - (8 - step)) return;
    
    pos_x = fmin(pos_x, xsize - 8);
    pos_y = fmin(pos_y, ysize - 8);
    
    double diff_xyb[3] = { 0.0 };
    Butteraugli8x8CornerEdgeDetectorDiff(pos_x, pos_y, xsize, ysize,
                                         r, g, b,
                                         r2, g2, b2,
                                         &diff_xyb[0]);
    
    int idx = (res_y * res_xsize + res_x) * 3;
    result[idx] = diff_xyb[0];
    result[idx + 1] = diff_xyb[1];
    result[idx + 2] = diff_xyb[2];
}

#endif
void clConvolutionEx(
                     cl_mem result/*out*/,
                     const cl_mem inp, size_t xsize, size_t ysize,
                     const cl_mem multipliers, size_t len,
                     int xstep, int offset, float border_ratio)
{
    ometal *m_ometal = [ometal sharedInstance];

    size_t oxsize = (xsize + xstep - 1) / xstep;

    id <MTLFunction> kernel =  m_ometal.kernel[KERNEL_CONVOLUTION];
    
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
    //        id <MTLFunction> kernel = [m_ometal.defaultLibrary newFunctionWithName:@"clConvolutionXEx"];
    NSError *errors;
    id <MTLComputePipelineState> filterState = [m_ometal.device newComputePipelineStateWithFunction:kernel error:&errors];
    
    [computeCE setComputePipelineState:filterState];
    [computeCE setBuffer:result offset:0 atIndex:0];
    [computeCE setBuffer:inp offset:0 atIndex:1];
    [computeCE setBuffer:xsizeBuffer offset:0 atIndex:2];
    [computeCE setBuffer:ysizeBuffer offset:0 atIndex:3];
    [computeCE setBuffer:multipliers offset:0 atIndex:4];
    [computeCE setBuffer:lenBuffer offset:0 atIndex:5];
    [computeCE setBuffer:xstepBuffer offset:0 atIndex:6];
    [computeCE setBuffer:offsetBuffer offset:0 atIndex:7];
    [computeCE setBuffer:border_ratioBuffer offset:0 atIndex:8];
    
    
    MTLSize threadsPerGroup = {1, 1, 1};
    MTLSize numThreadgroups= {xsize+10, ysize+10, 1};
    [computeCE dispatchThreadgroups:numThreadgroups
              threadsPerThreadgroup:threadsPerGroup];
    [computeCE endEncoding];
    
    // Commit the command buffer
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    //LOG_CL_RESULT(err);

}

void clConvolutionXEx(
                      cl_mem result/*out*/,
                      const cl_mem inp, size_t xsize, size_t ysize,
                      const cl_mem multipliers, size_t len,
                      int xstep, int offset, float border_ratio)
{
    ometal *m_ometal = [ometal sharedInstance];
    
//    static float inA[]={1.0,2.0,3.0,4.0};
//    static float inB[]={4.0,3.0,2.0,2.1};
//    static float out[]={0,0,0,0};
    //test
//        id<MTLBuffer> inABuffer = [m_ometal.device newBufferWithBytes:inA
//                                                  length:sizeof(inA)
//                                                 options:MTLResourceOptionCPUCacheModeDefault];
//        id<MTLBuffer> inBBuffer = [m_ometal.device newBufferWithBytes:inB
//                                                  length:sizeof(inB)
//                                                 options:MTLResourceOptionCPUCacheModeDefault];
//        id<MTLBuffer> outBuffer = [m_ometal.device newBufferWithBytes:out
//                                                  length:sizeof(out)
//                                                 options:MTLResourceOptionCPUCacheModeDefault];
//    
//        id <MTLCommandBuffer> commandBuffer1 = [m_ometal.commandQueue commandBuffer];
//    
//        // Create a compute command encoder
//        id <MTLComputeCommandEncoder> computeCE1 = [commandBuffer1 computeCommandEncoder];
//    
//    
//        NSError *errors1;
//    
//        id <MTLFunction> func = [m_ometal.defaultLibrary newFunctionWithName:@"kernel_function"];
//        id <MTLComputePipelineState> filterState1
//        = [m_ometal.device newComputePipelineStateWithFunction:func error:&errors1];
//    
//        [computeCE1 setComputePipelineState:filterState1];
//        [computeCE1 setBuffer:inABuffer offset:0 atIndex:0];
//        [computeCE1 setBuffer:inBBuffer offset:0 atIndex:1];
//        [computeCE1 setBuffer:outBuffer offset:0 atIndex:2];
//    
//    
//    
//        MTLSize threadsPerGroup1 = {2, 1, 1};
//        MTLSize numThreadgroups1 = {2, 1, 1};
//    
//        [computeCE1 dispatchThreadgroups:numThreadgroups1
//                  threadsPerThreadgroup:threadsPerGroup1];
//        [computeCE1 endEncoding];
//    
//        // Commit the command buffer
//        [commandBuffer1 commit];
//        [commandBuffer1 waitUntilCompleted];
//    
//        NSData* outdata = [NSData dataWithBytesNoCopy:[outBuffer contents ] length: sizeof(out) freeWhenDone:false ];
//    
//        float *values =(float *) [outdata bytes];
//        int cnt = [outdata length]/sizeof(float);
//        for (int i = 0; i < cnt; ++i)
//            NSLog(@"%f\n", values[i]);
//    
    
    //test end
    

    
    
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
    //        id <MTLFunction> kernel = [m_ometal.defaultLibrary newFunctionWithName:@"clConvolutionXEx"];
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
    MTLSize numThreadgroups= {xsize+10, ysize+10, 1};
    [computeCE dispatchThreadgroups:numThreadgroups
              threadsPerThreadgroup:threadsPerGroup];
    [computeCE endEncoding];
    
    // Commit the command buffer
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    //LOG_CL_RESULT(err);
    //err = clFinish(m_ometal.commandQueue);
    //LOG_CL_RESULT(err);

    
//    NSData* outdata = [NSData dataWithBytesNoCopy:[inp contents ] length: [inp length] freeWhenDone:false ];
//    NSLog(@"len: %d \n",[inp length]);
//    float *values =(float *) [outdata bytes];
//    int cnt = [outdata length]/sizeof(float);
//    for (int i = 0; i < cnt; ++i)
//        NSLog(@"%f\n", values[i]);
}

void clConvolutionYEx(
                      cl_mem result/*out*/,
                      const cl_mem inp, size_t xsize, size_t ysize,
                      const cl_mem multipliers, size_t len,
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
    //NSLog(@"clConvolutionYEx");
}

void clSquareSampleEx(
                      cl_mem result/*out*/,
                      const cl_mem image, size_t xsize, size_t ysize,
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
}

void clBlurEx(cl_mem image/*out, opt*/, const size_t xsize, const size_t ysize,
              const double sigma, const double border_ratio,
              cl_mem result/*out, opt*/)
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
    
    ometal *m_ometal = [ometal sharedInstance];
    cl_mem mem_expn = allocMem(sizeof(float) * expn_size, expn.data());
    
    if (xstep > 1)
    {
        cl_mem m = allocMem(sizeof(float) * xsize * ysize,NULL);
        clConvolutionXEx(m, image, xsize, ysize, mem_expn, expn_size, xstep, diff, border_ratio);
        clConvolutionYEx(result ? result : image, m, xsize, ysize, mem_expn, expn_size, xstep, diff, border_ratio);
        clSquareSampleEx(result ? result : image, result ? result : image, xsize, ysize, xstep, xstep);
        //clReleaseMemObject(m);
    }
    else
    {
        cl_mem m = allocMem(sizeof(float) * xsize * ysize,NULL);
        clConvolutionXEx(m, image, xsize, ysize, mem_expn, expn_size, xstep, diff, border_ratio);
        clConvolutionYEx(result ? result : image, m, xsize, ysize, mem_expn, expn_size, xstep, diff, border_ratio);
        //clReleaseMemObject(m);
    }
    
    // clReleaseMemObject(mem_expn);
    
    
}

void clOpsinDynamicsImageEx(ocl_channels  * rgb, const size_t xsize, const size_t ysize)
{
    static const double kSigma = 1.1;
    
    size_t channel_size = xsize * ysize * sizeof(float);
    
    ometal *m_ometal = [ometal sharedInstance];
    ocl_channels  * rgb_blurred = allocMemChannels(channel_size,NULL,NULL,NULL);
    
    const int size = xsize * ysize;
    
    clBlurEx(rgb->r, xsize, ysize, kSigma, 0.0, rgb_blurred->r);
    clBlurEx(rgb->g, xsize, ysize, kSigma, 0.0, rgb_blurred->g);
    clBlurEx(rgb->b, xsize, ysize, kSigma, 0.0, rgb_blurred->b);
    
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
    NSData* data = [NSData dataWithBytesNoCopy:[rgb->r contents ] length: [rgb->r length] freeWhenDone:false ];

}

void clMaskHighIntensityChangeEx(
                                 ocl_channels  * xyb0/*in,out*/,
                                 ocl_channels  * xyb1/*in,out*/,
                                 const size_t xsize, const size_t ysize)
{
    size_t channel_size = xsize * ysize * sizeof(float);
    
    ometal *m_ometal = [ometal sharedInstance];
    
    ocl_channels  * c0 = allocMemChannels(channel_size,NULL,NULL,NULL);
    ocl_channels  * c1 = allocMemChannels(channel_size,NULL,NULL,NULL);
    
    clEnqueueCopyBuffer(m_ometal.commandQueue, xyb0->r, c0->r, 0, 0, channel_size, 0, NULL, NULL);
    clEnqueueCopyBuffer(m_ometal.commandQueue, xyb0->g, c0->g, 0, 0, channel_size, 0, NULL, NULL);
    clEnqueueCopyBuffer(m_ometal.commandQueue, xyb0->b, c0->b, 0, 0, channel_size, 0, NULL, NULL);
    clEnqueueCopyBuffer(m_ometal.commandQueue, xyb1->r, c1->r, 0, 0, channel_size, 0, NULL, NULL);
    clEnqueueCopyBuffer(m_ometal.commandQueue, xyb1->g, c1->g, 0, 0, channel_size, 0, NULL, NULL);
    clEnqueueCopyBuffer(m_ometal.commandQueue, xyb1->b, c1->b, 0, 0, channel_size, 0, NULL, NULL);
    
    
    id <MTLFunction> kernel =  m_ometal.kernel[KERNEL_MASKHIGHINTENSITYCHANGE];
    //    clSetKernelArgEx(kernel,
    //                     &xyb0->r, &xyb0->g, &xyb0->b,
    //                     &xsize, &ysize,
    //                     &xyb1->r, &xyb1->g, &xyb1->b,
    //                     &c0.r, &c0.g, &c0.b,
    //                     &c1.r, &c1.g, &c1.b);
    //
    //    size_t globalWorkSize[2] = { xsize, ysize };
    //    int err = clEnqueueNDRangeKernel(m_ometal.commandQueue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
    //    LOG_CL_RESULT(err);
    //    err = clFinish(m_ometal.commandQueue);
    //    LOG_CL_RESULT(err);
    //
    //    m_m_ometal.releaseMemChannels(c0);
    //    m_ometal.releaseMemChannels(c1);
    //
    
    
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
    [computeCE setBuffer:c0->r offset:0 atIndex:8];
    [computeCE setBuffer:c0->g offset:0 atIndex:9];
    [computeCE setBuffer:c0->b offset:0 atIndex:10];
    [computeCE setBuffer:c1->r offset:0 atIndex:11];
    [computeCE setBuffer:c1->g offset:0 atIndex:12];
    [computeCE setBuffer:c1->b offset:0 atIndex:13];
    
    MTLSize threadsPerGroup = {1,1, 1};
    MTLSize numThreadgroups= {xsize, ysize, 1};
    [computeCE dispatchThreadgroups:numThreadgroups
              threadsPerThreadgroup:threadsPerGroup];
    [computeCE endEncoding];
    
    // Commit the command buffer
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
}

void clEdgeDetectorMapEx(
                         cl_mem result/*out*/,
                         const ocl_channels  * rgb, const ocl_channels  * rgb2,
                         const size_t xsize, const size_t ysize, const size_t step)
{

    size_t channel_size = xsize * ysize * sizeof(float);
    
    ometal *m_ometal = [ometal sharedInstance];
    
    ocl_channels  * rgb_blured = allocMemChannels(channel_size,NULL,NULL,NULL);
    ocl_channels  * rgb2_blured = allocMemChannels(channel_size,NULL,NULL,NULL);
    
    static const double kSigma[3] = { 1.5, 0.586, 0.4 };
    
    for (int i = 0; i < 3; i++)
    {
        clBlurEx(rgb->ch[i], xsize, ysize, kSigma[i], 0.0, rgb_blured->ch[i]);
        clBlurEx(rgb2->ch[i], xsize, ysize, kSigma[i], 0.0, rgb2_blured->ch[i]);
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
    NSData* data1 = [NSData dataWithBytesNoCopy:[result contents ] length: [result length] freeWhenDone:false ];
    //clEdgeDetectorMapEx((float *)[data1 bytes],&res_xsize,&res_ysize,rgb_blured->r,rgb_blured->g,rgb_blured->b,rgb2_blured->r,rgb2_blured->g,rgb2_blured->b,&xsize,&ysize,&step,0,0);
}

void clBlockDiffMapEx(
                      cl_mem block_diff_dc/*out*/,
                      cl_mem block_diff_ac/*out*/,
                      const ocl_channels  *rgb, const ocl_channels  *rgb2,
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
    MTLSize numThreadgroups = {res_xsize, res_xsize, 1};
    [computeCE dispatchThreadgroups:numThreadgroups
              threadsPerThreadgroup:threadsPerGroup];
    [computeCE endEncoding];
    
    // Commit the command buffer
    [commandBuffer commit];
    
}


void clEdgeDetectorLowFreqEx(
                             cl_mem block_diff_ac/*in,out*/,
                             const ocl_channels  *rgb, const ocl_channels  *rgb2,
                             const size_t xsize, const size_t ysize, const size_t step)
{
    size_t channel_size = xsize * ysize * sizeof(float);
    
    static const double kSigma = 14;
    ometal *m_ometal = [ometal sharedInstance];
    ocl_channels  * rgb_blured = allocMemChannels(channel_size,NULL,NULL,NULL);
    ocl_channels  * rgb2_blured = allocMemChannels(channel_size,NULL,NULL,NULL);
    
    for (int i = 0; i < 3; i++)
    {
        clBlurEx(rgb->ch[i], xsize, ysize, kSigma, 0.0, rgb_blured->ch[i]);
        clBlurEx(rgb2->ch[i], xsize, ysize, kSigma, 0.0, rgb2_blured->ch[i]);
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
}

void clDiffPrecomputeEx(
                        ocl_channels  *mask/*out*/,
                        const ocl_channels  *xyb0, const ocl_channels  *xyb1,
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
}

void clScaleImageEx(cl_mem img/*in, out*/, size_t size, double w)
{
    //NSData* data1 = [NSData dataWithBytesNoCopy:[img contents ] length: [img length] freeWhenDone:false ];
//    float *values =(float *) [data1 bytes];
//    int cnt = [data1 length]/sizeof(float);
//    for (int i = 0; i < 2; ++i)
//    NSLog(@"%f\n", values[i]);
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
    
    //NSData* data = [NSData dataWithBytesNoCopy:[img contents ] length: [img length] freeWhenDone:false ];
}

void clAverage5x5Ex(cl_mem img/*in,out*/, const size_t xsize, const size_t ysize)
{
    if (xsize < 4 || ysize < 4) {
        // TODO: Make this work for small dimensions as well.
        return;
    }
    
    ometal *m_ometal = [ometal sharedInstance];
    
    size_t len = xsize * ysize * sizeof(float);
    cl_mem img_org = allocMem(len,NULL);
    
    clEnqueueCopyBuffer(m_ometal.commandQueue, img, img_org, 0, 0, len, 0, NULL, NULL);
    
    id <MTLFunction> kernel =  m_ometal.kernel[KERNEL_AVERAGE5X5];
    //    clSetKernelArgEx(kernel, &img, &xsize, &ysize, &img_org);
    //
    //    size_t globalWorkSize[2] = { xsize, ysize };
    //    int err = clEnqueueNDRangeKernel(m_ometal.commandQueue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
    //    LOG_CL_RESULT(err);
    //    err = clFinish(m_ometal.commandQueue);
    //    LOG_CL_RESULT(err);
    //
    //    clReleaseMemObject(img_org);
    
    
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
}

void clMinSquareValEx(
                      cl_mem img/*in,out*/,
                      const size_t xsize, const size_t ysize,
                      const size_t square_size, const size_t offset)
{
    ometal *m_ometal = [ometal sharedInstance];
    
    cl_mem result = allocMem(sizeof(float) * xsize * ysize,NULL);
    
    id <MTLFunction> kernel =  m_ometal.kernel[KERNEL_MINSQUAREVAL];
    //    clSetKernelArgEx(kernel, &result, &xsize, &ysize, &img, &square_size, &offset);
    //
    //    size_t globalWorkSize[2] = { xsize, ysize };
    //    int err = clEnqueueNDRangeKernel(m_ometal.commandQueue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
    //    LOG_CL_RESULT(err);
    //    err = clEnqueueCopyBuffer(m_ometal.commandQueue, result, img, 0, 0, sizeof(float) * xsize * ysize, 0, NULL, NULL);
    //    LOG_CL_RESULT(err);
    //    err = clFinish(m_ometal.commandQueue);
    //    LOG_CL_RESULT(err);
    //    clReleaseMemObject(result);
    
    
    
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
    
    // Commit the command buffer
    [commandBuffer commit];
    
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

void clDoMask(ocl_channels  * mask/*in, out*/, ocl_channels  * mask_dc/*in, out*/, size_t xsize, size_t ysize)
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
    
    size_t channel_size = 512 * sizeof(double);
    ocl_channels  * xyb = allocMemChannels(channel_size, lut_x, lut_y, lut_b);
    ocl_channels  * xyb_dc = allocMemChannels(channel_size, lut_dcx, lut_dcy, lut_dcb);
    
    id <MTLFunction> kernel =  m_ometal.kernel[KERNEL_DOMASK];
    //    clSetKernelArgEx(kernel, &mask->r, &mask->g, &mask->b,
    //                     &xsize, &ysize,
    //                     &mask_dc->r, &mask_dc->g, &mask_dc->b,
    //                     &xyb.x, &xyb.y, &xyb.b,
    //                     &xyb_dc.x, &xyb_dc.y, &xyb_dc.b);
    //
    //    size_t globalWorkSize[2] = { xsize, ysize };
    //    int err = clEnqueueNDRangeKernel(m_ometal.commandQueue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
    //    LOG_CL_RESULT(err);
    //    err = clFinish(m_ometal.commandQueue);
    //    LOG_CL_RESULT(err);
    //
    //    m_ometal.releaseMemChannels(xyb);
    //    m_ometal.releaseMemChannels(xyb_dc);
    
    
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
}

void clMaskEx(
              ocl_channels  * mask/*out*/, ocl_channels  * mask_dc/*out*/,
              const ocl_channels  *rgb, const ocl_channels  *rgb2,
              const size_t xsize, const size_t ysize)
{

    clDiffPrecomputeEx(mask, rgb, rgb2, xsize, ysize);
    for (int i = 0; i < 3; i++)
    {
        clAverage5x5Ex(mask->ch[i], xsize, ysize);
        clMinSquareValEx(mask->ch[i], xsize, ysize, 4, 0);
        
        static const double sigma[3] = {
            9.65781083553,
            14.2644604355,
            4.53358927369,
        };
        
        clBlurEx(mask->ch[i], xsize, ysize, sigma[i], 0.0);
    }
    
    clDoMask(mask, mask_dc, xsize, ysize);
    NSData* data1 = [NSData dataWithBytesNoCopy:[mask->r contents ] length: [mask->r length] freeWhenDone:false ];
    for (int i = 0; i < 3; i++)
    {
        clScaleImageEx(mask->ch[i], xsize * ysize, kGlobalScale * kGlobalScale);
        clScaleImageEx(mask_dc->ch[i], xsize * ysize, kGlobalScale * kGlobalScale);
    }
}

void clCombineChannelsEx(
                         cl_mem result/*out*/,
                         const ocl_channels  *mask,
                         const ocl_channels  *mask_dc,
                         const size_t xsize, const size_t ysize,
                         const cl_mem block_diff_dc,
                         const cl_mem block_diff_ac,
                         const cl_mem edge_detector_map,
                         const size_t res_xsize,
                         const size_t step)
{
    
    
    
    NSData* data = [NSData dataWithBytesNoCopy:[mask->r contents ] length: [mask->r length] freeWhenDone:false ];
    NSData* data1 = [NSData dataWithBytesNoCopy:[mask_dc->r contents ] length: [mask_dc->r length] freeWhenDone:false ];

    ometal *m_ometal = [ometal sharedInstance];
    
    const size_t work_xsize = ((xsize - 8 + step) + step - 1) / step;
    const size_t work_ysize = ((ysize - 8 + step) + step - 1) / step;
    
    id <MTLFunction> kernel =  m_ometal.kernel[KERNEL_COMBINECHANNELS];
    //    clSetKernelArgEx(kernel, &result,
    //                     &mask->r, &mask->g, &mask->b,
    //                     &mask_dc->r, &mask_dc->g, &mask_dc->b,
    //                     &xsize, &ysize,
    //                     &block_diff_dc, &block_diff_ac,
    //                     &edge_detector_map,
    //                     &res_xsize,
    //                     &step);
    //
    //    size_t globalWorkSize[2] = { work_xsize, work_ysize };
    //    int err = clEnqueueNDRangeKernel(m_ometal.commandQueue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
    //    LOG_CL_RESULT(err);
    //    err = clFinish(m_ometal.commandQueue);
    //    LOG_CL_RESULT(err);
    
    
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
    MTLSize numThreadgroups = {xsize, ysize, 1};
    [computeCE dispatchThreadgroups:numThreadgroups
              threadsPerThreadgroup:threadsPerGroup];
    [computeCE endEncoding];
    
    // Commit the command buffer
    [commandBuffer commit];
    NSData* data2 = [NSData dataWithBytesNoCopy:[result contents ] length: [result length] freeWhenDone:false ];

}

void clUpsampleSquareRootEx(cl_mem diffmap, const size_t xsize, const size_t ysize, const int step)
{
    
    
    
    ometal *m_ometal = [ometal sharedInstance];
    
    cl_mem diffmap_out = allocMem(xsize * ysize * sizeof(float),NULL);
    
    id <MTLFunction> kernel =  m_ometal.kernel[KERNEL_UPSAMPLESQUAREROOT];
    //    clSetKernelArgEx(kernel, &diffmap_out, &diffmap, &xsize, &ysize, &step);
    
    const size_t res_xsize = (xsize + step - 1) / step;
    const size_t res_ysize = (ysize + step - 1) / step;
    
    //    size_t globalWorkSize[2] = { res_xsize, res_ysize };
    //    int err = clEnqueueNDRangeKernel(m_ometal.commandQueue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
    //    LOG_CL_RESULT(err);
    //
    //    LOG_CL_RESULT(err);
    //    err = clFinish(m_ometal.commandQueue);
    //    LOG_CL_RESULT(err);
    //
    //    clReleaseMemObject(diffmap_out);
    
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
    [computeCE setBuffer:diffmap offset:0 atIndex:1];
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
    
    
    
    //NSData* data = [NSData dataWithBytesNoCopy:[diffmap_out contents ] length: [diffmap_out length] freeWhenDone:false ];
    //NSData* data1 = [NSData dataWithBytesNoCopy:[diffmap contents ] length: [diffmap length] freeWhenDone:false ];
    clEnqueueCopyBuffer(m_ometal.commandQueue, diffmap_out, diffmap, 0, 0, xsize * ysize * sizeof(float), 0, NULL, NULL);
}

void clRemoveBorderEx(cl_mem out, const cl_mem in, const size_t xsize, const size_t ysize, const int step)
{
    ometal *m_ometal = [ometal sharedInstance];
    
    int cls = 8 - step;
    int cls2 = (8 - step) / 2;
    
    int out_xsize = xsize - cls;
    int out_ysize = ysize - cls;
    
    id <MTLFunction> kernel =  m_ometal.kernel[KERNEL_REMOVEBORDER];
    //    clSetKernelArgEx(kernel, &out, &out_xsize, &out_ysize, &in, &cls, &cls2);
    //
    //    size_t globalWorkSize[2] = { out_xsize, out_ysize};
    //    int err = clEnqueueNDRangeKernel(m_ometal.commandQueue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
    //    LOG_CL_RESULT(err);
    //    err = clFinish(m_ometal.commandQueue);
    //    LOG_CL_RESULT(err);
    
    
    id <MTLCommandBuffer> commandBuffer = [m_ometal.commandQueue commandBuffer];
    
    // Create a compute command encoder
    id <MTLComputeCommandEncoder> computeCE = [commandBuffer computeCommandEncoder];
    id<MTLBuffer> out_xsizeBuffer =[m_ometal.device newBufferWithBytes:&out_xsize
                                                                length:sizeof(&out_xsize)
                                                               options:MTLResourceOptionCPUCacheModeDefault];
    id<MTLBuffer> out_ysizeBuffer =[m_ometal.device newBufferWithBytes:&out_ysize
                                                                length:sizeof(&out_ysize)
                                                               options:MTLResourceOptionCPUCacheModeDefault];
    id<MTLBuffer> clsBuffer =[m_ometal.device newBufferWithBytes:&cls
                                                          length:sizeof(&cls)
                                                         options:MTLResourceOptionCPUCacheModeDefault];
    id<MTLBuffer> cls2Buffer =[m_ometal.device newBufferWithBytes:&cls2
                                                           length:sizeof(&cls2)
                                                          options:MTLResourceOptionCPUCacheModeDefault];
    NSError *errors;
    id <MTLComputePipelineState> filterState
    = [m_ometal.device newComputePipelineStateWithFunction:kernel error:&errors];
    
    
    [computeCE setComputePipelineState:filterState];
    [computeCE setBuffer:out offset:0 atIndex:0];
    [computeCE setBuffer:out_xsizeBuffer offset:0 atIndex:1];
    [computeCE setBuffer:out_ysizeBuffer offset:0 atIndex:2];
    [computeCE setBuffer:in offset:0 atIndex:3];
    [computeCE setBuffer:clsBuffer offset:0 atIndex:4];
    [computeCE setBuffer:cls2Buffer offset:0 atIndex:5];
    
    
    
    MTLSize threadsPerGroup = {1,1, 1};
    MTLSize numThreadgroups = {static_cast<NSUInteger>(out_xsize), static_cast<NSUInteger>(out_ysize), 1};
    [computeCE dispatchThreadgroups:numThreadgroups
              threadsPerThreadgroup:threadsPerGroup];
    [computeCE endEncoding];
    
    // Commit the command buffer
    [commandBuffer commit];
}

void clAddBorderEx(cl_mem out, size_t xsize, size_t ysize, int step, cl_mem in)
{
    ometal *m_ometal = [ometal sharedInstance];
    
    int cls = 8 - step;
    int cls2 = (8 - step) / 2;
    id <MTLFunction> kernel =  m_ometal.kernel[KERNEL_ADDBORDER];
    //    clSetKernelArgEx(kernel, &out, &xsize, &ysize, &cls, &cls2, &in);
    //
    //    size_t globalWorkSize[2] = { xsize, ysize};
    //    int err = clEnqueueNDRangeKernel(m_ometal.commandQueue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
    //    LOG_CL_RESULT(err);
    //    err = clFinish(m_ometal.commandQueue);
    //    LOG_CL_RESULT(err);
    
    
    id <MTLCommandBuffer> commandBuffer = [m_ometal.commandQueue commandBuffer];
    
    // Create a compute command encoder
    id <MTLComputeCommandEncoder> computeCE = [commandBuffer computeCommandEncoder];
    id<MTLBuffer> xsizeBuffer =[m_ometal.device newBufferWithBytes:&xsize
                                                            length:sizeof(&xsize)
                                                           options:MTLResourceOptionCPUCacheModeDefault];
    id<MTLBuffer> ysizeBuffer =[m_ometal.device newBufferWithBytes:&ysize
                                                            length:sizeof(&ysize)
                                                           options:MTLResourceOptionCPUCacheModeDefault];
    id<MTLBuffer> clsBuffer =[m_ometal.device newBufferWithBytes:&cls
                                                          length:sizeof(&cls)
                                                         options:MTLResourceOptionCPUCacheModeDefault];
    id<MTLBuffer> cls2Buffer =[m_ometal.device newBufferWithBytes:&cls2
                                                           length:sizeof(&cls2)
                                                          options:MTLResourceOptionCPUCacheModeDefault];
    NSError *errors;
    id <MTLComputePipelineState> filterState
    = [m_ometal.device newComputePipelineStateWithFunction:kernel error:&errors];
    
    
    [computeCE setComputePipelineState:filterState];
    [computeCE setBuffer:out offset:0 atIndex:0];
    [computeCE setBuffer:xsizeBuffer offset:0 atIndex:1];
    [computeCE setBuffer:ysizeBuffer offset:0 atIndex:2];
    [computeCE setBuffer:clsBuffer offset:0 atIndex:3];
    [computeCE setBuffer:cls2Buffer offset:0 atIndex:4];
    [computeCE setBuffer:in offset:0 atIndex:5];
    
    
    
    MTLSize threadsPerGroup = {1,1, 1};
    MTLSize numThreadgroups = {xsize, ysize, 1};
    [computeCE dispatchThreadgroups:numThreadgroups
              threadsPerThreadgroup:threadsPerGroup];
    [computeCE endEncoding];
    
    // Commit the command buffer
    [commandBuffer commit];
    
    //NSData* data = [NSData dataWithBytesNoCopy:[out contents ] length: [out length] freeWhenDone:false ];
}

void clCalculateDiffmapEx(cl_mem diffmap/*in,out*/, const size_t xsize, const size_t ysize, const int step)
{
    clUpsampleSquareRootEx(diffmap, xsize, ysize, step);
    
    static const double kSigma = 8.8510880283;
    static const double mul1 = 24.8235314874;
    static const double scale = 1.0 / (1.0 + mul1);
    
    const int s = 8 - step;
    int s2 = (8 - step) / 2;
    
    ometal *m_ometal = [ometal sharedInstance];
    cl_mem blurred = allocMem((xsize - s) * (ysize - s) * sizeof(float),NULL);
    clRemoveBorderEx(blurred, diffmap, xsize, ysize, step);
    
    static const double border_ratio = 0.03027655136;
    clBlurEx(blurred, xsize - s, ysize - s, kSigma, border_ratio);
    
    clAddBorderEx(diffmap, xsize, ysize, step, blurred);
    clScaleImageEx(diffmap, xsize * ysize, scale);
    
    
}
#ifdef __USE_DOUBLE_AS_FLOAT__
#undef double
#endif

@interface metalguezli ()





@end
@implementation metalguezli

@end
