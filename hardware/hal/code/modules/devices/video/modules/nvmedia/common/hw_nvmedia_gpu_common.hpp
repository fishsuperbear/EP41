#ifndef HW_NVMEDIA_GPU_COMMON_H
#define HW_NVMEDIA_GPU_COMMON_H


#define ALIGN_STEP 512
#define ALIGN(width) ((width+ALIGN_STEP-1)&(~(ALIGN_STEP-1)))
namespace netaos {
    namespace gpu {
        enum class DataType : unsigned int{
            None             = 0,
            Uint8            = 1,
            Float32          = 2,
            Float16          = 3
        };

    } /* namespace gpu */
} /* namespace netaos */
typedef unsigned char uint8_t;

enum class PixelLayout : unsigned int{
    None       = 0,
    NCHW_RGB   = 1,  //RRR...GGG...BBB...
    NCHW_BGR   = 2,
    NHWC_RGB   = 3,
    NHWC_BGR   = 4,
    NCHW16_RGB = 5,  // c = (c + 15) / 16 * 16 if c % 16 != 0 else c
    NCHW16_BGR = 6   // NHW16 if c == 3
};
enum GPU_IMG_TYPE{
    DEFAULT,
    GPU_Bayer_RGB888 = DEFAULT,
    GPU_YUV420_NV12,
    GPU_YUV422_YUYV,
};


enum class Interpolation : unsigned int{
    None     = 0,
    Nearest  = 1,
    Bilinear = 2
};

enum class YUVFormat : unsigned int{
    None                            = 0,
    NV12BlockLinear                 = 1,
    NV12PitchLinear                 = 2,
    YUV422Packed_YUYV_PitchLinear   = 3
};

// NV12 YUV PL
struct NV12HostImage{
    uint8_t* data = nullptr;
    unsigned int width  = 0, height = 0;
    unsigned int y_area = 0;
    unsigned int stride = 0;
    YUVFormat format  = YUVFormat::None;
};

typedef struct YUVGPUImage{
    void* luma   = 0;     // y
    void* chroma = 0;     // uv
    void* luma_array   = nullptr;        //  nullptr if format == PL
    void* chroma_array = nullptr;        //  nullptr if format == PL
    unsigned int width = 0, height = 0;
    unsigned int stride = 0;
    unsigned int batch = 0;
    YUVFormat format  = YUVFormat::None;
} YUVGPUImage;

typedef struct RGBGPUImage {
	void* data  = nullptr;
	int width = 0;
	int height = 0;
	int stride = 0;
	int batch   = 0;
    int channel = 0;
    PixelLayout layout = PixelLayout::None;
    netaos::gpu::DataType dtype     = netaos::gpu::DataType::None;
} RGBGPUImage;

typedef struct GPUImage{
    GPU_IMG_TYPE out_img_type=GPU_IMG_TYPE::DEFAULT;
    void* image = nullptr;
} GPUImage;

struct FillColor{unsigned char color[3];};

#endif //HW_NVMEDIA_GPU_COMMON_H
