#ifndef CAMERADEVICEIMPL_HPP
#define CAMERADEVICEIMPL_HPP

#include "CameraDeviceSessionImpl.hpp"

/*
 * Currently, one camera device can only has one session.
 */
#define CAMERADEVICE_SESSION_MAXNUM         3
/*
 * Currently, one camera device can only has one hw_video_t ops.
 */
#define CAMERADEVICE_HW_VIDEO_MAXNUM        3

class CameraDeviceImplEnv;
struct VideoProcessor{
    hw_module_t* pmodule;
    hw_video_t* pvideo;
};
/*
 * All lower level ops like hw_video_t should be get and maintain in the class while other class only
 * get the pointer but not put or maintain it.
 */
class CameraDeviceImpl : public ICameraDevice
{
    friend class CameraDeviceSessionImpl;
    public:
    virtual s32 CreateCameraSession(ICameraDeviceCallback* i_pcallback, ICameraDeviceSession** o_ppsession);
    /*
     * Within mutex_global lock.
     */
    virtual s32 RegisterEventCallback(CameraDeviceEventCbRegInfo* i_peventreginfo, camera_device_eventcb i_eventcb);
    virtual s32 RegisterCpuDataCallback(const CameraDeviceCpuDataCbRegInfo* i_pdatareginfo, camera_device_datacb i_datacb);
    virtual s32 RegisterGpuDataCallback(const CameraDeviceGpuDataCbRegInfo* i_pgpureginfo, camera_device_gpudatacb i_gpudatacb);
    /*
     * Currently, you should do all of the release operation like Close before you delete the
     * CameraDeviceImpl instance.
     */
    virtual ~CameraDeviceImpl();
    protected:
    /*
     * The operation should be called internally by CameraDeviceSessionImpl::Close function.
     * The session instance will not delete, it will be reinit when next use.
     */
    s32 Internal_CloseSession(u32 i_sessionindex);
    protected:
    static void Internal_LowerLevelDataCb(struct hw_video_bufferinfo_t* i_pbufferinfo);
    static void Internal_LowerLevelGpuDataCb(struct hw_video_cudabufferinfo_t* i_pcudabufferinfo);
    protected:
    static bool IsRGBType_ByGPUDataCbImgType(CAMERA_DEVICE_GPUDATACB_IMGTYPE i_imgtype);
    static bool IsYUVType_ByGPUDataCbImgType(CAMERA_DEVICE_GPUDATACB_IMGTYPE i_imgtype);
    static s32 GetLowerLevelBlockIndex_ByOpenTypeBlockType(CAMERA_DEVICE_OPENTYPE i_opentype,
            CAMERA_DEVICE_BLOCK_TYPE i_blocktype, u32* o_pblockindex);
    static s32 GetLowerLevelBlockSensorIndex_ByOpenTypeBlockSensorIndex(CAMERA_DEVICE_OPENTYPE i_opentype,
            CAMERA_DEVICE_BLOCK_TYPE i_blocktype, u32 i_sensorindex, u32* o_pblockindex, u32* o_psensorindex);
    static s32 GetLowerLevelDataCbType_ByDataCbType(CAMERA_DEVICE_DATACB_TYPE i_datacbtype, HW_VIDEO_REGDATACB_TYPE* o_plowerleveldatacbtype);
    static s32 GetLowerLevelCudaImgType_ByGpuImageType(CAMERA_DEVICE_GPUDATACB_IMGTYPE i_gpuimgtype, HW_VIDEO_REGCUDADATACB_IMGTYPE* o_plowerlevelcudaimgtype);
    static s32 GetLowerLevelInterpolation_ByInterpolation(CAMERA_DEVICE_GPUDATACB_INTERPOLATION i_interpolation, HW_VIDEO_REGCUDADATACB_INTERPOLATION* o_plowerlevelinterpolation);
    protected:
    /*
     * Currently only one instance, so only one session counter sessionnum.
     * Currently sessionnum should be at most 1.
     */
    u32                             _sessionnum = 0;
    /*
     * Currently only one hw_video_t, so only one hw_video_t counter videonum.
     * Currently videonum should be at most 1.
     */
    u32                             _videonum = 0;
    /*
     * You must set it to nullptr when init.
     */
    CameraDeviceSessionImpl*        _parray_psession[CAMERADEVICE_SESSION_MAXNUM] = { nullptr };
    /*
     * The operation of the _psession_toinit is within mutex_global lock.
     */
    CameraDeviceSessionImpl*        _psession_toinit = nullptr;
    VideoProcessor _pvideoprocessors[CAMERADEVICE_SESSION_MAXNUM];
    ICameraDeviceSession* _sub_session = nullptr;
};

/*
 * The _mutex_global correspondent structure.
 */
struct CameraDeviceImplGlobalEnv
{
    /*
     * After GetInstance it is set to 1, when release it is set back to 0.
     */
    u32                             bhasinstance = 0;
    /*
     * We store the camera device instance here for bak use.
     */
    CameraDeviceImpl*               pcameradevice = nullptr;

};


class CameraDeviceImplEnv
{
    public:
        CameraDeviceImplEnv();
        ~CameraDeviceImplEnv();
    public:
        u32                             binit = 0;
        struct hw_mutex_t               mutex_global;
        CameraDeviceImplGlobalEnv       globalenv;

};
#endif
