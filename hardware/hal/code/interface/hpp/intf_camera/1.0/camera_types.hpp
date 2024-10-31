#ifndef CAMERA_TYPES_HPP
#define CAMERA_TYPES_HPP

#include <iostream>
#include <cstring>
#include <iostream>
#include <csignal>
#include <thread>
#include <chrono>
#include <ctime>
#include <atomic>
#include <cmath>
#include <fstream>
#include <vector>
#include <map>
#include <memory>

#ifndef s8
typedef signed char         s8;
#endif
#ifndef s16
typedef short               s16;
#endif
#ifndef s32
typedef int                 s32;
#endif
#ifndef s64
typedef long long           s64;
#endif
#ifndef u8
typedef unsigned char       u8;
#endif
#ifndef u16
typedef unsigned short      u16;
#endif
#ifndef u32
typedef unsigned int        u32;
#endif
#ifndef u64
typedef unsigned long long  u64;
#endif

#ifndef STATIC_ASSERT
#define STATIC_ASSERT(truecond)     static_assert (truecond, "error")
#endif

STATIC_ASSERT(sizeof(void*) == 8);

STATIC_ASSERT(sizeof(u8) == 1);
STATIC_ASSERT(sizeof(u16) == 2);
STATIC_ASSERT(sizeof(u32) == 4);
STATIC_ASSERT(sizeof(u64) == 8);

/*
 * Define the header file version here!
 */
enum HAL_CAMERA_VERSION : u32 {
    /*
     * The current version.
     * Version 1.0
     */
    HAL_CAMERA_VERSION_1_0 = 1,
};


enum CAMERA_DEVICE_BLOCK_TYPE : u32 {
    CAMERA_DEVICE_BLOCK_TYPE_GROUPA = 1,
    CAMERA_DEVICE_BLOCK_TYPE_GROUPB = 2,
    CAMERA_DEVICE_BLOCK_TYPE_GROUPC = 3,
    CAMERA_DEVICE_BLOCK_TYPE_GROUPD = 4,
};

enum CAMERA_DEVICE_OPENTYPE : u32 {
    //open all sensor desay
    CAMERA_DEVICE_OPENTYPE_MULTIROUP_SENSOR_DESAY = 1,
};

/*
 * The query info of the camera device sensor info of specific opentype and specific block type.
 * CameraDeviceOpenTypeSensorInfo may be inherited in the later version.
 */
class CameraDeviceOpenTypeSensorInfo
{
    public:
        CAMERA_DEVICE_OPENTYPE              opentype;
        CAMERA_DEVICE_BLOCK_TYPE            blocktype;
        u32                                 capturewidth;
        u32                                 captureheight;
};

enum CAMERA_DEVICE_OPENMODE : u32 {
    // it decide how to config the pipeline
    CAMERA_DEVICE_OPENMODE_MAIN = 1,
    // it only get data or wait to get data, it will not config or start the pipeline
    CAMERA_DEVICE_OPENMODE_SUB = 2,
};

enum CAMERA_DEVICE_EVENTTYPE : u32 {
    CAMERA_DEVICE_EVENTTYPE_MSG_DEVICE      = 1,
    CAMERA_DEVICE_EVENTTYPE_MSG_FRAME     = 2,
};

enum CAMERA_DEVICE_MSGCODE : u32 {
    /*--Frame Msg--*/
    MSG_FRAME_READY = 1,
    MSG_FRAME_DROP,
    /*--Device Msg--*/
};

class CameraDeviceMsgDevice
{
    public:
        CAMERA_DEVICE_MSGCODE           msgtype;
};

class CameraDeviceMsgFrame
{
    public:
        CAMERA_DEVICE_MSGCODE           msgtype;
        u64                             timestamp;
};

class CameraDeviceEventCbRegInfo
{
    public:
        CAMERA_DEVICE_OPENTYPE              opentype;
        CAMERA_DEVICE_BLOCK_TYPE            blocktype;
        // begin from 0, of the specific sensortype of the specific blocktype
        u32                                 sensorindex;
};

/*
 * CameraDeviceCpuDataCbRegInfo may be inherited in the later version.
 */
class CameraDeviceEventCbInfo
{
    public:
        CAMERA_DEVICE_OPENTYPE              opentype;
        CAMERA_DEVICE_BLOCK_TYPE            blocktype;
        // begin from 0, of the specific sensortype of the specific blocktype
        u32                                 sensorindex;
        CAMERA_DEVICE_EVENTTYPE             eventtype;
        /*
         * The registered custom pointer which will transfer back when data callback.
         */
        void*                               pcontext;
        /*
         * Define two variables for extending event callback information
         *      if (eventtype == CAMERA_DEVICE_EVENTTYPE_MSG_DEVICE)
         *          pvoid1 = CameraDeviceMsgFrame*
         *      elif (eventtype == CAMERA_DEVICE_EVENTTYPE_MSG_FRAME)
         *          pvoid1 = CameraDeviceMsgDevice*
         */
        void*                               pvoid1;
        void*                               pvoid2;
};

enum CAMERA_DEVICE_DATACB_TYPE : u32 {
    CAMERA_DEVICE_DATACB_TYPE_RAW12 = 1,
    CAMERA_DEVICE_DATACB_TYPE_YUV422 = 2,
    // common yuv420 format
    CAMERA_DEVICE_DATACB_TYPE_YUV420 = 3,
    // private yuv420 format
    CAMERA_DEVICE_DATACB_TYPE_YUV420_PRIV = 4,
    CAMERA_DEVICE_DATACB_TYPE_RGBA = 5,
    CAMERA_DEVICE_DATACB_TYPE_AVC = 6,
    CAMERA_DEVICE_DATACB_TYPE_HEVC = 7,
};

/*
 * The register info of the camera device data callback type.
 * CameraDeviceCpuDataCbRegInfo may be inherited in the later version.
 */
class CameraDeviceCpuDataCbRegInfo
{
    public:
        CAMERA_DEVICE_OPENTYPE              opentype;
        CAMERA_DEVICE_BLOCK_TYPE            blocktype;
        // begin from 0, of the specific sensortype of the specific blocktype
        u32                                 sensorindex;
        CAMERA_DEVICE_DATACB_TYPE           datacbtype;
        /*
         * 0 or 1
         * 1 means use the origin capture width and height.
         */
        u32                                 busecaptureresolution;
        /*
         * Valid when busecaptureresolution is 0.
         * It can be the same as the origin capture width.
         */
        u32                                 customwidth;
        /*
         * Valid when busecaptureresolution is 0.
         * It can be the same as the origin capture height.
         */
        u32                                 customheight;
        /*
         * Use the default origin captured frame rate.
         */
        u32                                 busecaptureframerate;
        /*
         * Valid when busecaptureframerate is 0.
         * The user setting frame rates per minute.
         */
        u32                                 customframerate;
        /*
         * Rotate clockwise degrees. 45 multiple.
         */
        u32                                 rotatedegrees;
        /*
         * The registered custom pointer which will transfer back when data callback.
         */
        void*                               pcontext;
};

enum CAMERA_DEVICE_GPUDATACB_IMGTYPE : u32 {
    // rgb888 sub type
    CAMERA_DEVICE_GPUDATACB_IMGTYPE_RGB888_NHWC_RGB = 1,
    // rgb888 sub type, the common rgb888 sub type
    CAMERA_DEVICE_GPUDATACB_IMGTYPE_RGB888_NHWC_BGR = 2,
    // rgb888 sub type
    CAMERA_DEVICE_GPUDATACB_IMGTYPE_RGB888_NCHW_RGB = 3,
    // rgb888 sub type
    CAMERA_DEVICE_GPUDATACB_IMGTYPE_RGB888_NCHW_BGR = 4,
    // rgb888 sub type
    CAMERA_DEVICE_GPUDATACB_IMGTYPE_RGB888_NCHW16_RGB = 5,
    // rgb888 sub type
    CAMERA_DEVICE_GPUDATACB_IMGTYPE_RGB888_NCHW16_BGR = 6,
    // yuv420 sub type, the common yuv420 sub type
    CAMERA_DEVICE_GPUDATACB_IMGTYPE_NV12 = 7,
    // yuv420 sub type
    CAMERA_DEVICE_GPUDATACB_IMGTYPE_NV21 = 8,
    // yuv422 sub type, the common yuv422 sub type
    CAMERA_DEVICE_GPUDATACB_IMGTYPE_YUYV = 9,
    // yuv422 sub type
    CAMERA_DEVICE_GPUDATACB_IMGTYPE_YVYU = 10,
    // yuv422 sub type
    CAMERA_DEVICE_GPUDATACB_IMGTYPE_VYUY = 11,
};

enum CAMERA_DEVICE_GPUDATACB_INTERPOLATION : u32 {
    // Nearest cost time less but effect may not be so good
    CAMERA_DEVICE_GPUDATACB_INTERPOLATION_NEAREST = 1,
    // Bilinear cost time but effect may be good
    CAMERA_DEVICE_GPUDATACB_INTERPOLATION_BILINEAR = 2,
};

class CameraDeviceGpuDataCbRegInfo
{
    public:
        CAMERA_DEVICE_OPENTYPE              opentype;
        CAMERA_DEVICE_BLOCK_TYPE            blocktype;
        // begin from 0, of the specific sensortype of the specific blocktype
        u32                                 sensorindex;
        CAMERA_DEVICE_GPUDATACB_IMGTYPE     gpuimgtype;
        CAMERA_DEVICE_GPUDATACB_INTERPOLATION   interpolation;
        /*
         * 0 or 1
         * 1 means use the origin capture width and height.
         */
        u32                                 busecaptureresolution;
        /*
         * Valid when busecaptureresolution is 0.
         * It can be the same as the origin capture width.
         */
        u32                                 customwidth;
        /*
         * Valid when busecaptureresolution is 0.
         * It can be the same as the origin capture height.
         */
        u32                                 customheight;
        /*
         * Use the default origin captured frame rate.
         */
        u32                                 busecaptureframerate;
        /*
         * Valid when busecaptureframerate is 0.
         * The user setting frame rates per minute.
         */
        u32                                 customframerate;
        /*
         * Rotate clockwise degrees. 45 multiple.
         */
        u32                                 rotatedegrees;
        /*
         * The registered custom pointer which will transfer back when data callback.
         */
        void*                               pcontext;
};

struct CameraDeviceDataCbTimeInfo
{
    u64                                 timestamp;
};

/*
 * Context pointer inside.
 * CameraDeviceDataCbInfo may be inherited in the later version.
 */
class CameraDeviceDataCbInfo
{
    public:
        /*
         * The time we receive the frame.
         * Currently, we use the monotonic system time.
         */
        CameraDeviceDataCbTimeInfo          timeinfo;
        CAMERA_DEVICE_OPENTYPE              opentype;
        CAMERA_DEVICE_BLOCK_TYPE            blocktype;
        // begin from 0, of the specific sensortype
        u32                                 sensorindex;
        /*
         * Same as the input parameter i_type when you call ICameraDevice::RegisterDataCallback.
         */
        CAMERA_DEVICE_DATACB_TYPE           datacbtype;
        /*
         * The origin capture width.
         */
        u32                                 capturewidth;
        /*
         * The origin capture height.
         */
        u32                                 captureheight;
        /*
         * The width of the pbuff frame buffer.
         */
        u32                                 width;
        /*
         * The height of the pbuff frame buffer.
         */
        u32                                 height;
        /*
         * The stride for width alignment use.
         * Use 0 to mean no stride. 512 step in common use.
         * The stride sentence sample is like the following:
         * #define ALIGN_STEP 512
         * #define ALIGN(width) ((width+ALIGN_STEP-1)&(~(ALIGN_STEP-1)))
         */
        u32                                 stride;
        /*
         * Rotate clockwise degrees. 0-360.
         */
        u32                                 rotatedegrees;
        /*
         * The frame buffer start pointer.
         */
        void*                               pbuff;
        /*
         * The frame buffer size.
         */
        u32                                 size;
        /*
         * The custom pointer when register the data callback.
         */
        void*                               pcontext;
};

/*
 * YUV data callback information.
 * CameraDeviceGpuDataCbGpuBuffYUVInfo may be inherited in the later version.
 */
class CameraDeviceGpuDataCbGpuBuffYUVInfo
{
    public:
        /*
         * YUV types only. Reuse img type defines.
         */
        CAMERA_DEVICE_GPUDATACB_IMGTYPE     imgtype;
        void*                               plumabuff;
        u32                                 lumabuffsize;
        void*                               pchromabuff;
        u32                                 chromabuffsize;
        /*
         * Bak use for private yuv format.
         */
        void*                               plumabuff_priv;
        // size of plumabuff_priv byte count
        u32                                 lumabuffprivsize;
        /*
         * Bak use for private yuv format.
         */
        void*                               pchromabuff_priv;
        // size of pchromabuff_priv byte count
        u32                                 chromabuffprivsize;
};

/*
 * YUV data callback information.
 * CameraDeviceGpuDataCbGpuBuffRGBInfo may be inherited in the later version.
 */
class CameraDeviceGpuDataCbGpuBuffRGBInfo
{
    public:
        /*
         * RGB types only. Reuse img type defines.
         */
        CAMERA_DEVICE_GPUDATACB_IMGTYPE     imgtype;
        void*                               pbuff;
        u32                                 buffsize;
};

class CameraDeviceGpuDataCbGpuBuffInfo
{
    public:
        CAMERA_DEVICE_GPUDATACB_IMGTYPE         rgbtype;
        /*
         * Is nullptr when is not yuv format.
         */
        CameraDeviceGpuDataCbGpuBuffYUVInfo*    pyuvinfo;
        /*
         * Is nullptr when is not rgb format.
         */
        CameraDeviceGpuDataCbGpuBuffRGBInfo*    prgbinfo;
};

/*
 * Context pointer inside.
 * CameraDeviceGpuDataCbInfo may be inherited in the later version.
 */
class CameraDeviceGpuDataCbInfo
{
    public:
        /*
         * The time we receive the frame.
         * Currently, we use the monotonic system time.
         */
        CameraDeviceDataCbTimeInfo          timeinfo;
        CAMERA_DEVICE_OPENTYPE              opentype;
        CAMERA_DEVICE_BLOCK_TYPE            blocktype;
        // begin from 0, of the specific sensortype
        u32                                 sensorindex;
        CAMERA_DEVICE_GPUDATACB_IMGTYPE     gpuimgtype;
        CAMERA_DEVICE_GPUDATACB_INTERPOLATION   interpolation;
        /*
         * The origin capture width.
         */
        u32                                 capturewidth;
        /*
         * The origin capture height.
         */
        u32                                 captureheight;
        /*
         * The width of the pbuff gpu frame buffer.
         */
        u32                                 width;
        /*
         * The height of the pbuff gpu frame buffer.
         */
        u32                                 height;
        /*
         * The stride for width alignment use.
         * Use 0 to mean no stride. 512 step in common use.
         * The stride sentence sample is like the following:
         * #define ALIGN_STEP 512
         * #define ALIGN(width) ((width+ALIGN_STEP-1)&(~(ALIGN_STEP-1)))
         */
        u32                                 stride;
        /*
         * Rotate clockwise degrees. 0-360.
         */
        u32                                 rotatedegrees;
        /*
         * The gpu buffer start pointer.
         */
        CameraDeviceGpuDataCbGpuBuffInfo*   pgpuinfo;
        /*
         * The custom pointer when register the data callback.
         */
        void*                               pcontext;
};

/*
 * CameraDeviceDataCbInfo has context pointer inside.
 * CameraDeviceDataCbInfo may be inherited in the later version.
 */
typedef void(*camera_device_eventcb)(CameraDeviceEventCbInfo* i_peventcbinfo);
/*
 * CameraDeviceDataCbInfo has context pointer inside.
 * CameraDeviceDataCbInfo may be inherited in the later version.
 */
typedef void(*camera_device_datacb)(CameraDeviceDataCbInfo* i_pdatacbinfo);
/*
 * CameraDeviceGpuDataCbInfo has context pointer inside.
 * CameraDeviceGpuDataCbInfo may be inherited in the later version.
 */
typedef void(*camera_device_gpudatacb)(CameraDeviceGpuDataCbInfo* i_pgpudatacbinfo);

#endif
