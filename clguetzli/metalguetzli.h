//
//  metalguetzli.hpp
//  guetzli_ios
//
//  Created by 张聪 on 2017/9/13.
//  Copyright © 2017年 com.tencent. All rights reserved.
//
#ifdef __USE_METAL__


#import <Foundation/Foundation.h>
#import "ometal.h"
#import "guetzli/processor.h"
#ifdef __USE_DOUBLE_AS_FLOAT__
#define double float
#endif

#import "butteraugli_comparator.h"

enum MATH_MODE
{
    MODE_CPU = 6,
    MODE_CPU_OPT = 1,
    MODE_OPENCL =2,
    MODE_CUDA=3,
    MODE_CHECKCL=4,
    MODE_CHECKCUDA=5,
    MODE_METAL=0,
    MODE_CHECKMETAL=7,
};

static MATH_MODE g_mathMode = MODE_METAL;

void metalEnqueueReadBuffer(id     /* command_queue */,
                            metal_mem m_metal_mem             /* buffer */,
                            bool             /* blocking_read */,
                            size_t              /* offset */,
                            size_t    size          /* size */,
                            void *   m_ptr           /* ptr */,
                            uint             /* num_events_in_wait_list */,
                            void *    /* event_wait_list */,
                            void *          /* event */);

void metalOpsinDynamicsImage(
                             float *r, float *g, float *b,
                             const size_t xsize, const size_t ysize);

void metalDiffmapOpsinDynamicsImage(
                                    float* result,
                                    const float* r, const float* g, const float* b,
                                    const float* r2, const float* g2, const float* b2,
                                    const size_t xsize, const size_t ysize,
                                    const size_t step);

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
                                   const float BlockErrorLimit);

void metalMask(
               float* mask_r, float* mask_g, float* mask_b,
               float* maskdc_r, float* maskdc_g, float* maskdc_b,
               const size_t xsize, const size_t ysize,
               const float* r, const float* g, const float* b,
               const float* r2, const float* g2, const float* b2);

void metalDiffmapOpsinDynamicsImageEx(
                                      metal_mem result,
                                      ometal_channels  *     xyb0,
                                      ometal_channels  *     xyb1,
                                      const size_t xsize, const size_t ysize,
                                      const size_t step);

void metalConvolutionXEx(
                         metal_mem result/*out*/,
                         const metal_mem inp, size_t xsize, size_t ysize,
                         const metal_mem multipliers, size_t len,
                         int xstep, int offset, float border_ratio);

void metalConvolutionYEx(
                         metal_mem result/*out*/,
                         const metal_mem inp, size_t xsize, size_t ysize,
                         const metal_mem multipliers, size_t len,
                         int xstep, int offset, float border_ratio);

void metalSquareSampleEx(
                         metal_mem result/*out*/,
                         const metal_mem image, size_t xsize, size_t ysize,
                         size_t xstep, size_t ystep);

void metalBlurEx(metal_mem image/*out, opt*/, const size_t xsize, const size_t ysize,
                 const double sigma, const double border_ratio,
                 metal_mem result = NULL/*out, opt*/);

void metalOpsinDynamicsImageEx(   ometal_channels  *     rgb, const size_t xsize, const size_t ysize);

void metalMaskHighIntensityChangeEx(
                                    ometal_channels  *   xyb0/*in,out*/,
                                    ometal_channels  *   xyb1/*in,out*/,
                                    const size_t xsize, const size_t ysize);

void metalEdgeDetectorMapEx(
                            metal_mem result/*out*/,
                            const  ometal_channels  *   rgb, const  ometal_channels  *   rgb2,
                            const size_t xsize, const size_t ysize, const size_t step);

void metalBlockDiffMapEx(
                         metal_mem block_diff_dc/*out*/,
                         metal_mem block_diff_ac/*out*/,
                         const  ometal_channels  *   rgb, const  ometal_channels  *   rgb2,
                         const size_t xsize, const size_t ysize, const size_t step);

void metalEdgeDetectorLowFreqEx(
                                metal_mem block_diff_ac/*in,out*/,
                                const   ometal_channels  *     rgb, const   ometal_channels  *     rgb2,
                                const size_t xsize, const size_t ysize, const size_t step);

void metalDiffPrecomputeEx(
                           ometal_channels  *     mask/*out*/,
                           const    ometal_channels  * xyb0, const   ometal_channels  * xyb1,
                           const size_t xsize, const size_t ysize);

void metalScaleImageEx(metal_mem img/*in, out*/, size_t size, double w);

void metalAverage5x5Ex(metal_mem img/*in,out*/, const size_t xsize, const size_t ysize);

metal_mem metalMinSquareValEx(
                              metal_mem img/*in,out*/,
                              const size_t xsize, const size_t ysize,
                              const size_t square_size, const size_t offset,int unuse);

void metalMaskEx(
                 ometal_channels  *       mask/*out*/,     ometal_channels  *   mask_dc/*out*/,
                 const    ometal_channels  *     rgb, const      ometal_channels  *  rgb2,
                 const size_t xsize, const size_t ysize);

void metalCombineChannelsEx(
                            metal_mem result/*out*/,
                            const   ometal_channels  * mask,
                            const   ometal_channels  * mask_dc,
                            const size_t xsize, const size_t ysize,
                            const metal_mem block_diff_dc,
                            const metal_mem block_diff_ac,
                            const metal_mem edge_detector_map,
                            const size_t res_xsize,
                            const size_t step);

void metalUpsampleSquareRootEx(metal_mem *diffmap, const size_t xsize, const size_t ysize, const int step);

void metalRemoveBorderEx(metal_mem out, const metal_mem in, const size_t xsize, const size_t ysize, const int step);

void metalAddBorderEx(metal_mem out, const size_t xsize, const size_t ysize, const int step, const metal_mem in);

void metalCalculateDiffmapEx(metal_mem *diffmap/*in,out*/, const size_t xsize, const size_t ysize, const int step);

namespace guetzli {
    
    class ButteraugliComparatorEx : public ButteraugliComparator
    {
    public:
        ButteraugliComparatorEx(const int width, const int height,
                                const std::vector<uint8_t>* rgb,
                                const float target_distance, ProcessStats* stats);
        
        void Compare(const OutputImage& img) override;
        void StartBlockComparisons() override;
        void FinishBlockComparisons() override;
        double CompareBlock(const OutputImage& img, int off_x, int off_y, const coeff_t* candidate_block, const int comp_mask) const override;
        
    public:
        std::vector<float> imgOpsinDynamicsBlockList;   // [RR..RRGG..GGBB..BB]:blockCount
        std::vector<float> imgMaskXyzScaleBlockList;    // [RGBRGB..RGBRGB]:blockCount
        std::vector<std::vector<float>> rgb_orig_opsin;
    };
}

#ifdef __USE_DOUBLE_AS_FLOAT__
#undef double
#endif


#endif

