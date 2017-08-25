#include <algorithm>
#include <stdint.h>
#include <vector>
#include "utils.h"

#import "ometal.h"
#ifdef __USE_METAL__


using namespace std;
#define __device__
#define __private
#define __constant_ex
#define __constant


static __constant double g_gamma_q[5 + 1] = {
    12.262350348616792, 20.557285797683576, 12.161463238367844,
    4.711532733641639, 0.899112889751053, 0.035662329617191,
};

static float g_mix[12] = {
    0.348036746003,
    0.577814843137,
    0.0544556093735,
    0.774145581713,
    0.26922717275,
    0.767247733938,
    0.0366922708552,
    0.920130265014,
    0.0882062883536,
    0.158581714673,
    0.712857943858,
    10.6524069248,
};
static  __constant double g_gamma_p[5 + 1] = {
    881.979476556478289, 1496.058452015812463, 908.662212739659481,
    373.566100223287378, 85.840860336314364, 6.683258861509244,
};

__device__ void CalcOpsinDynamicsImage(__private float rgb[3][(8 * 8)]);
__device__ void OpsinDynamicsImageBlock(__private float *r, __private float *g, __private float *b,
                                        __private const float *r_blurred, __private const float *g_blurred, __private const float *b_blurred,
                                        int size);
__device__ void OpsinAbsorbance(const double in[3], double out[3]);
__device__ void RgbToXyb(double r, double g, double b, double *valx, double *valy, double *valz);
__device__ double Gamma(double v);
__device__ double EvaluatePolynomial(const double x, __constant_ex const double *coefficients, int n);
__device__ void BlurEx(const float *r, int xsize, int ysize, double kSigma, double border_ratio, float *output);
__device__ void Convolution(size_t xsize, size_t ysize,
                            int xstep, int len, int offset,
                            const float* multipliers,
                            const float* inp,
                            float border_ratio,
                            float* result);

static double kSrgb8ToLinearTable[256] = {
    0.000000,
    0.077399,
    0.154799,
    0.232198,
    0.309598,
    0.386997,
    0.464396,
    0.541796,
    0.619195,
    0.696594,
    0.773994,
    0.853367,
    0.937509,
    1.026303,
    1.119818,
    1.218123,
    1.321287,
    1.429375,
    1.542452,
    1.660583,
    1.783830,
    1.912253,
    2.045914,
    2.184872,
    2.329185,
    2.478910,
    2.634105,
    2.794824,
    2.961123,
    3.133055,
    3.310673,
    3.494031,
    3.683180,
    3.878171,
    4.079055,
    4.285881,
    4.498698,
    4.717556,
    4.942502,
    5.173584,
    5.410848,
    5.654341,
    5.904108,
    6.160196,
    6.422649,
    6.691512,
    6.966827,
    7.248640,
    7.536993,
    7.831928,
    8.133488,
    8.441715,
    8.756651,
    9.078335,
    9.406810,
    9.742115,
    10.084290,
    10.433375,
    10.789410,
    11.152432,
    11.522482,
    11.899597,
    12.283815,
    12.675174,
    13.073712,
    13.479465,
    13.892470,
    14.312765,
    14.740385,
    15.175366,
    15.617744,
    16.067555,
    16.524833,
    16.989614,
    17.461933,
    17.941824,
    18.429322,
    18.924460,
    19.427272,
    19.937793,
    20.456054,
    20.982090,
    21.515934,
    22.057618,
    22.607175,
    23.164636,
    23.730036,
    24.303404,
    24.884774,
    25.474176,
    26.071642,
    26.677203,
    27.290891,
    27.912736,
    28.542769,
    29.181020,
    29.827520,
    30.482299,
    31.145387,
    31.816813,
    32.496609,
    33.184802,
    33.881422,
    34.586499,
    35.300062,
    36.022139,
    36.752760,
    37.491953,
    38.239746,
    38.996169,
    39.761248,
    40.535013,
    41.317491,
    42.108710,
    42.908697,
    43.717481,
    44.535088,
    45.361546,
    46.196882,
    47.041124,
    47.894297,
    48.756429,
    49.627547,
    50.507676,
    51.396845,
    52.295078,
    53.202402,
    54.118843,
    55.044428,
    55.979181,
    56.923129,
    57.876298,
    58.838712,
    59.810398,
    60.791381,
    61.781686,
    62.781338,
    63.790363,
    64.808784,
    65.836627,
    66.873918,
    67.920679,
    68.976937,
    70.042715,
    71.118037,
    72.202929,
    73.297414,
    74.401516,
    75.515259,
    76.638668,
    77.771765,
    78.914575,
    80.067122,
    81.229428,
    82.401518,
    83.583415,
    84.775142,
    85.976722,
    87.188178,
    88.409534,
    89.640813,
    90.882037,
    92.133229,
    93.394412,
    94.665609,
    95.946841,
    97.238133,
    98.539506,
    99.850982,
    101.172584,
    102.504334,
    103.846254,
    105.198366,
    106.560693,
    107.933256,
    109.316077,
    110.709177,
    112.112579,
    113.526305,
    114.950375,
    116.384811,
    117.829635,
    119.284868,
    120.750532,
    122.226647,
    123.713235,
    125.210317,
    126.717914,
    128.236047,
    129.764737,
    131.304005,
    132.853871,
    134.414357,
    135.985483,
    137.567270,
    139.159738,
    140.762907,
    142.376799,
    144.001434,
    145.636832,
    147.283012,
    148.939997,
    150.607804,
    152.286456,
    153.975971,
    155.676371,
    157.387673,
    159.109900,
    160.843070,
    162.587203,
    164.342319,
    166.108438,
    167.885578,
    169.673761,
    171.473005,
    173.283330,
    175.104755,
    176.937299,
    178.780982,
    180.635824,
    182.501843,
    184.379058,
    186.267489,
    188.167154,
    190.078073,
    192.000265,
    193.933749,
    195.878543,
    197.834666,
    199.802137,
    201.780975,
    203.771198,
    205.772826,
    207.785876,
    209.810367,
    211.846319,
    213.893748,
    215.952674,
    218.023115,
    220.105089,
    222.198615,
    224.303711,
    226.420395,
    228.548685,
    230.688599,
    232.840156,
    235.003373,
    237.178269,
    239.364861,
    241.563167,
    243.773205,
    245.994993,
    248.228549,
    250.473890,
    252.731035,
    255.000000,
};
//#define __checkcl
//#define abs(exper)    fabs((exper))
//#include "clguetzli.h"
//#include "clguetzli.cl"
//#include "cuguetzli.h"
//#include "ocu.h"

#import "metalguezli.h"
#import "ometal.h"

namespace guetzli
{
    ButteraugliComparatorEx::ButteraugliComparatorEx(const int width, const int height,
        const std::vector<uint8_t>* rgb,
        const float target_distance, ProcessStats* stats)
        : ButteraugliComparator(width, height, rgb, target_distance, stats)
    {
        if (MODE_CPU != g_mathMode)
        {
            rgb_orig_opsin.resize(3);
            rgb_orig_opsin[0].resize(width * height);
            rgb_orig_opsin[1].resize(width * height);
            rgb_orig_opsin[2].resize(width * height);

#ifdef __USE_DOUBLE_AS_FLOAT__
            const float* lut = kSrgb8ToLinearTable;
#else
            const double* lut = kSrgb8ToLinearTable;
#endif
            for (int c = 0; c < 3; ++c) {
                for (int y = 0, ix = 0; y < height_; ++y) {
                    for (int x = 0; x < width_; ++x, ++ix) {
                        rgb_orig_opsin[c][ix] = lut[rgb_orig_[3 * ix + c]];
                    }
                }
            }
            ::butteraugli::OpsinDynamicsImage(width_, height_, rgb_orig_opsin);
        }
    }

    void ButteraugliComparatorEx::Compare(const OutputImage& img)
    {
		if (MODE_CPU_OPT == g_mathMode)
		{
			std::vector<std::vector<float> > rgb0 = rgb_orig_opsin;

			std::vector<std::vector<float> > rgb(3, std::vector<float>(width_ * height_));
			img.ToLinearRGB(&rgb);
			::butteraugli::OpsinDynamicsImage(width_, height_, rgb);
			std::vector<float>().swap(distmap_);
			comparator_.DiffmapOpsinDynamicsImage(rgb0, rgb, distmap_);
			distance_ = ::butteraugli::ButteraugliScoreFromDiffmap(distmap_);
		}
        
#ifdef __USE_METAL__
        else if (MODE_METAL == g_mathMode)
        {
            std::vector<std::vector<float> > rgb1(3, std::vector<float>(width_ * height_));
            img.ToLinearRGB(&rgb1);
            
            const int xsize = width_;
            const int ysize = height_;
            std::vector<float>().swap(distmap_);
            distmap_.resize(xsize * ysize);
            
            size_t channel_size = xsize * ysize * sizeof(float);
            
            ometal *m_ometal = [ometal sharedInstance];
            
            cl_mem mem_result = allocMem(channel_size,NULL);
            ocl_channels *xyb0 = allocMemChannels(channel_size, rgb_orig_opsin[0].data(), rgb_orig_opsin[1].data(), rgb_orig_opsin[2].data());
            ocl_channels *xyb1 = allocMemChannels(channel_size, rgb1[0].data(), rgb1[1].data(), rgb1[2].data());
            clOpsinDynamicsImageEx(xyb1, xsize, ysize);
            clDiffmapOpsinDynamicsImageEx(mem_result, xyb0, xyb1, xsize, ysize, comparator_.step());

            clEnqueueReadBuffer(m_ometal.commandQueue, mem_result, false, 0, channel_size, distmap_.data(), 0, NULL, NULL);
            
            
            
            distance_ = ::butteraugli::ButteraugliScoreFromDiffmap(distmap_);
        }
#endif
#ifdef __USE_OPENCL__
        else if (MODE_OPENCL == g_mathMode)
        {
            std::vector<std::vector<float> > rgb1(3, std::vector<float>(width_ * height_));
            img.ToLinearRGB(&rgb1);

            const int xsize = width_;
            const int ysize = height_;
            std::vector<float>().swap(distmap_);
            distmap_.resize(xsize * ysize);

            size_t channel_size = xsize * ysize * sizeof(float);
            ocl_args_d_t &ocl = getOcl();
            ocl_channels xyb0 = ocl.allocMemChannels(channel_size, rgb_orig_opsin[0].data(), rgb_orig_opsin[1].data(), rgb_orig_opsin[2].data());
            ocl_channels xyb1 = ocl.allocMemChannels(channel_size, rgb1[0].data(), rgb1[1].data(), rgb1[2].data());

            cl_mem mem_result = ocl.allocMem(channel_size);

            clOpsinDynamicsImageEx(xyb1, xsize, ysize);
            clDiffmapOpsinDynamicsImageEx(mem_result, xyb0, xyb1, xsize, ysize, comparator_.step());

            cl_int err = clEnqueueReadBuffer(ocl.commandQueue, mem_result, false, 0, channel_size, distmap_.data(), 0, NULL, NULL);
            LOG_CL_RESULT(err);
            err = clFinish(ocl.commandQueue);
            LOG_CL_RESULT(err);

            clReleaseMemObject(mem_result);
            ocl.releaseMemChannels(xyb0);
            ocl.releaseMemChannels(xyb1);

            distance_ = ::butteraugli::ButteraugliScoreFromDiffmap(distmap_);
        }
#endif
#ifdef __USE_CUDA__
        else if (MODE_CUDA == g_mathMode)
        {
            std::vector<std::vector<float> > rgb1(3, std::vector<float>(width_ * height_));
            img.ToLinearRGB(&rgb1);

            const int xsize = width_;
            const int ysize = height_;
            std::vector<float>().swap(distmap_);
            distmap_.resize(xsize * ysize);

            size_t channel_size = xsize * ysize * sizeof(float);
            ocu_args_d_t &ocu = getOcu();
            ocu_channels xyb0 = ocu.allocMemChannels(channel_size, rgb_orig_opsin[0].data(), rgb_orig_opsin[1].data(), rgb_orig_opsin[2].data());
            ocu_channels xyb1 = ocu.allocMemChannels(channel_size, rgb1[0].data(), rgb1[1].data(), rgb1[2].data());
            
            cu_mem mem_result = ocu.allocMem(channel_size);

            cuOpsinDynamicsImageEx(xyb1, xsize, ysize);

            cuDiffmapOpsinDynamicsImageEx(mem_result, xyb0, xyb1, xsize, ysize, comparator_.step());

            cuMemcpyDtoH(distmap_.data(), mem_result, channel_size);

            ocu.releaseMem(mem_result);
            ocu.releaseMemChannels(xyb0);
            ocu.releaseMemChannels(xyb1);

            distance_ = ::butteraugli::ButteraugliScoreFromDiffmap(distmap_);
        }
#endif
		else
		{
			ButteraugliComparator::Compare(img);
		}
    }

    void ButteraugliComparatorEx::StartBlockComparisons()
    {
        if (MODE_CPU == g_mathMode)
        {
            ButteraugliComparator::StartBlockComparisons();
            return;
        }

        std::vector<std::vector<float> > dummy(3);
        ::butteraugli::Mask(rgb_orig_opsin, rgb_orig_opsin, width_, height_, &mask_xyz_, &dummy);

        const int width = width_;
        const int height = height_;
        const int factor_x = 1;
        const int factor_y = 1;

        const int block_width = (width + 8 * factor_x - 1) / (8 * factor_x);
        const int block_height = (height + 8 * factor_y - 1) / (8 * factor_y);
        const int num_blocks = block_width * block_height;
#ifdef __USE_DOUBLE_AS_FLOAT__
        const float* lut = kSrgb8ToLinearTable;
#else
        const double* lut = kSrgb8ToLinearTable;
#endif
        imgOpsinDynamicsBlockList.resize(num_blocks * 3 * (8 * 8));
        imgMaskXyzScaleBlockList.resize(num_blocks * 3);
        for (int block_y = 0, block_ix = 0; block_y < block_height; ++block_y)
        {
            for (int block_x = 0; block_x < block_width; ++block_x, ++block_ix)
            {
                float* curR = &imgOpsinDynamicsBlockList[block_ix * 3 * (8 * 8)];
                float* curG = curR + (8 * 8);
                float* curB = curG + (8 * 8);

                for (int iy = 0, i = 0; iy < 8; ++iy) {
                    for (int ix = 0; ix < 8; ++ix, ++i) {
                        int x = std::min(8 * block_x + ix, width - 1);
                        int y = std::min(8 * block_y + iy, height - 1);
                        int px = y * width + x;

                        curR[i] = lut[rgb_orig_[3 * px]];
                        curG[i] = lut[rgb_orig_[3 * px + 1]];
                        curB[i] = lut[rgb_orig_[3 * px + 2]];
                    }
                }

                CalcOpsinDynamicsImage((float(*)[64])curR);

                int xmin = block_x * 8;
                int ymin = block_y * 8;

                imgMaskXyzScaleBlockList[block_ix * 3] = mask_xyz_[0][ymin * width_ + xmin];
                imgMaskXyzScaleBlockList[block_ix * 3 + 1] = mask_xyz_[1][ymin * width_ + xmin];
                imgMaskXyzScaleBlockList[block_ix * 3 + 2] = mask_xyz_[2][ymin * width_ + xmin];
            }
        }
    }

    void ButteraugliComparatorEx::FinishBlockComparisons() {
        ButteraugliComparator::FinishBlockComparisons();

        imgOpsinDynamicsBlockList.clear();
        imgMaskXyzScaleBlockList.clear();
    }
    
    double ButteraugliComparatorEx::CompareBlock(const OutputImage& img, int off_x, int off_y, const coeff_t* candidate_block, const int comp_mask) const
    {
        double err = ButteraugliComparator::CompareBlock(img, off_x, off_y, candidate_block, comp_mask);

        return err;
    }
}




__device__ void CalcOpsinDynamicsImage(__private float rgb[3][(8 * 8)])
{
    float rgb_blurred[3][(8 * 8)];
    for (int i = 0; i < 3; i++)
    {
        BlurEx(rgb[i], 8, 8, 1.1, 0, rgb_blurred[i]);
    }
    OpsinDynamicsImageBlock(rgb[0], rgb[1], rgb[2], rgb_blurred[0], rgb_blurred[1], rgb_blurred[2], (8 * 8));
}


__device__ void OpsinDynamicsImageBlock(__private float *r, __private float *g, __private float *b,
                                        __private const float *r_blurred, __private const float *g_blurred, __private const float *b_blurred,
                                        int size)
{
    for (size_t i = 0; i < size; ++i) {
        double sensitivity[3];
        {
            // Calculate sensitivity[3] based on the smoothed image gamma derivative.
            double pre_rgb[3] = { r_blurred[i], g_blurred[i], b_blurred[i] };
            double pre_mixed[3];
            OpsinAbsorbance(pre_rgb, pre_mixed);
            sensitivity[0] = Gamma(pre_mixed[0]) / pre_mixed[0];
            sensitivity[1] = Gamma(pre_mixed[1]) / pre_mixed[1];
            sensitivity[2] = Gamma(pre_mixed[2]) / pre_mixed[2];
        }
        double cur_rgb[3] = { r[i],  g[i],  b[i] };
        double cur_mixed[3];
        OpsinAbsorbance(cur_rgb, cur_mixed);
        cur_mixed[0] *= sensitivity[0];
        cur_mixed[1] *= sensitivity[1];
        cur_mixed[2] *= sensitivity[2];
        double x, y, z;
        RgbToXyb(cur_mixed[0], cur_mixed[1], cur_mixed[2], &x, &y, &z);
        r[i] = (float)(x);
        g[i] = (float)(y);
        b[i] = (float)(z);
    }
}

__device__ void OpsinAbsorbance(const double in[3], double out[3])
{
    out[0] = g_mix[0] * in[0] + g_mix[1] * in[1] + g_mix[2] * in[2] + g_mix[3];
    out[1] = g_mix[4] * in[0] + g_mix[5] * in[1] + g_mix[6] * in[2] + g_mix[7];
    out[2] = g_mix[8] * in[0] + g_mix[9] * in[1] + g_mix[10] * in[2] + g_mix[11];
}

__device__ void RgbToXyb(double r, double g, double b, double *valx, double *valy, double *valz)
{
    const double a0 = 1.01611726948;
    const double a1 = 0.982482243696;
    const double a2 = 1.43571362627;
    const double a3 = 0.896039849412;
    *valx = a0 * r - a1 * g;
    *valy = a2 * r + a3 * g;
    *valz = b;
}

__device__ double Gamma(double v)
{
    const double min_value = 0.770000000000000;
    const double max_value = 274.579999999999984;
    const double x01 = (v - min_value) / (max_value - min_value);
    const double xc = 2.0 * x01 - 1.0;
    
    const double yp = EvaluatePolynomial(xc, g_gamma_p, 6);
    const double yq = EvaluatePolynomial(xc, g_gamma_q, 6);
    if (yq == 0.0) return 0.0;
    return (float)(yp / yq);
}

__device__ double EvaluatePolynomial(const double x, __constant_ex const double *coefficients, int n)
{
    double b1 = 0.0;
    double b2 = 0.0;
    
    for (int i = n - 1; i >= 0; i--)
    {
        if (i == 0) {
            const double x_b1 = x * b1;
            b1 = x_b1 - b2 + coefficients[0];
            break;
        }
        const double x_b1 = x * b1;
        const double t = (x_b1 + x_b1) - b2 + coefficients[i];
        b2 = b1;
        b1 = t;
    }
    
    return b1;
}

__device__ void BlurEx(const float *r, int xsize, int ysize, double kSigma, double border_ratio, float *output)
{
    const double sigma = 1.1;
    double m = 2.25;  // Accuracy increases when m is increased.
    const double scaler = -0.41322314049586772; // when sigma=1.1, scaler is -0.41322314049586772
    const int diff = 2;  // when sigma=1.1, diff's value is 2.
    const int expn_size = 5; // when sigma=1.1, scaler is  5
    float expn[5] = { static_cast<float>(exp(scaler * (-diff) * (-diff))),
        static_cast<float>(exp(scaler * (-diff + 1) * (-diff + 1))),
        static_cast<float>(exp(scaler * (-diff + 2) * (-diff + 2))),
        static_cast<float>(exp(scaler * (-diff + 3) * (-diff + 3))),
        static_cast<float>(exp(scaler * (-diff + 4) * (-diff + 4)))};
    const int xstep = 1; // when sigma=1.1, xstep is 1.
    const int ystep = xstep;
    
    int dxsize = (xsize + xstep - 1) / xstep;
    
    float tmp[8*8] = { 0 };
    Convolution(xsize, ysize, xstep, expn_size, diff, expn, r, border_ratio, tmp);
    Convolution(ysize, dxsize, ystep, expn_size, diff, expn, tmp,
                border_ratio, output);
}


__device__ void Convolution(size_t xsize, size_t ysize,
                            int xstep, int len, int offset,
                            const float* multipliers,
                            const float* inp,
                            float border_ratio,
                            float* result)
{
    float weight_no_border = 0;
    
    for (size_t j = 0; j <= 2 * offset; ++j) {
        weight_no_border += multipliers[j];
    }
    for (size_t x = 0, ox = 0; x < xsize; x += xstep, ox++) {
        int minx = x < offset ? 0 : x - offset;
        int maxx = min(xsize, x + len - offset) - 1;
        float weight = 0.0;
        for (int j = minx; j <= maxx; ++j) {
            weight += multipliers[j - x + offset];
        }
        // Interpolate linearly between the no-border scaling and border scaling.
        weight = (1.0 - border_ratio) * weight + border_ratio * weight_no_border;
        float scale = 1.0 / weight;
        for (size_t y = 0; y < ysize; ++y) {
            float sum = 0.0;
            for (int j = minx; j <= maxx; ++j) {
                sum += inp[y * xsize + j] * multipliers[j - x + offset];
            }
            result[ox * ysize + y] = (float)(sum * scale);
        }
    }
}

#endif
