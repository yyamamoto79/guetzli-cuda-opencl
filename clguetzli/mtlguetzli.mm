//
//  mtlguetzli.m
//  guetzli_osx
//
//  Created by pine wang on 17/7/10.
//  Copyright © 2017年 com.tencent. All rights reserved.
//
#import "mtlguetzli.h"
#import <Foundation/Foundation.h>
#import "processor.h"
#import "mtlguetzli.metal"

bool testProcess(const guetzli::Params& params, guetzli::ProcessStats* stats,
                 const std::string& in_data,
                 std::string* out_data){
    
    return NO;
}


//void mtlDiffmapOpsinDynamicsImage(
//                                 float* result,
//                                 const float* r,  const float* g,  const float* b,
//                                 const float* r2, const float* g2, const float* b2,
//                                 const size_t xsize, const size_t ysize,
//                                 const size_t step)
//{
//    size_t channel_size = xsize * ysize * sizeof(float);
//    
//    ocl_args_d_t &ocl = getOcl();
//    ocl_channels xyb0 = ocl.allocMemChannels(channel_size, r, g, b);
//    ocl_channels xyb1 = ocl.allocMemChannels(channel_size, r2, g2, b2);
//    
//    cl_mem mem_result = ocl.allocMem(channel_size, result);
//    
//    clDiffmapOpsinDynamicsImageEx(mem_result, xyb0, xyb1, xsize, ysize, step);
//    
//    clEnqueueReadBuffer(ocl.commandQueue, mem_result, false, 0, channel_size, result, 0, NULL, NULL);
//    cl_int err = clFinish(ocl.commandQueue);
//    
//    ocl.releaseMemChannels(xyb1);
//    ocl.releaseMemChannels(xyb0);
//    
//    clReleaseMemObject(mem_result);
//}
