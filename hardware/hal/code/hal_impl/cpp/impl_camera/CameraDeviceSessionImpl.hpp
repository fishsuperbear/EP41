#ifndef CAMERADEVICESESSIONIMPL_HPP
#define CAMERADEVICESESSIONIMPL_HPP

#include "hal_camera_log_impl.h"

struct CameraDeviceDataCbEnv
{
    /*
    * 0 or 1.
    * 1 when register data type is gpu callback.
    * 0 when register data type is not gpu callback.
    */
    u32                             bgpucb;
    /*
    * Valid when bgpucb is 0.
    * The structure when register data callback.
    */
    CameraDeviceCpuDataCbRegInfo        datacbreginfo;
    /*
    * Valid when bgpucb is 0.
    * The data callback set by user.
    */
    camera_device_datacb            datacb;
    /*
    * Valid when bgpucb is 0.
    * The base structure used by data callback input parameter.
    * We need to copy it to a temp stack variable and change its buffer pointer
    * and size and then transfer to the camera_device_datacb datacb.
    */
    CameraDeviceDataCbInfo          bufferinfo;
    /*
    * Valid when bgpucb is 1.
    * The structure when register gpu data callback.
    */
    CameraDeviceGpuDataCbRegInfo    gpudatacbreginfo;
    /*
    * Valid when bgpucb is 1.
    * The gpu data callback set by user.
    */
    camera_device_gpudatacb         gpudatacb;
    /*
    * Valid when bgpucb is 1.
    * The base structure used by gpu data callback input parameter.
    * We need to copy it to a temp stack variable and change its buffer pointer
    * and size and then transfer to the camera_device_gpudatacb gpudatacb.
    */
    CameraDeviceGpuDataCbInfo       gpubufferinfo;
    /*
    * The instance of the pointer which belongs to CameraDeviceGpuDataCbInfo.
    */
    CameraDeviceGpuDataCbGpuBuffInfo    gpubufferinfo_buffinfo;
    /*
    * The instance of the pointer which belongs to CameraDeviceGpuDataCbGpuBuffInfo.
    */
    CameraDeviceGpuDataCbGpuBuffYUVInfo gpubufferinfo_yuvinfo;
    /*
    * The instance of the pointer which belongs to CameraDeviceGpuDataCbGpuBuffInfo.
    */
    CameraDeviceGpuDataCbGpuBuffRGBInfo gpubufferinfo_rgbinfo;
    /*
    * Lower level needed data, blockindex.
    */
    u32                             lowerlevelblockindex;
    /*
    * Lower level needed data, sensorindex.
    */
    u32                             lowerlevelsensorindex;
    /*
    * Lower level needed HW_VIDEO_REGDATACB_TYPE.
    */
    HW_VIDEO_REGDATACB_TYPE         lowerlevelregdatacbtype;
    /*
    * The lower level data callback used by lower level ops.
    */
    hw_video_sensorpipelinedatacb   lowerleveldatacb;
    /*
    * The lower level cuda data callback used by lower level ops.
    */
    hw_video_sensorpipelinecudadatacb   lowerlevelcudadatacb;
};
enum SessionType{
    SESSION_TYPE_UNKNOW = -1,
    SESSION_TYPE_DESAY_MAIN = 0,
    SESSION_TYPE_DESAY_SUB_SHM_ENC,
    SESSION_TYPE_DESAY_SUB_CUDA
};

/*
* Used by lower level pipeline data callback.
* The input parameter structure is defined by lower level, you cannnot change it.
*/
void Internal_LowerLevelDataCb(struct hw_video_bufferinfo_t* i_pbufferinfo);
/*
* Used by lower level pipeline gpu data callback.
* The input parameter structure is defined by lower level, you cannnot change it.
*/
void Internal_LowerLevelGpuDataCb(struct hw_video_cudabufferinfo_t* i_pcudabufferinfo);

class CameraDeviceImpl;

class CameraDeviceSessionImpl : public ICameraDeviceSession
{
    friend class CameraDeviceImpl;
public:
    CameraDeviceSessionImpl() = delete;
    CameraDeviceSessionImpl(u32 i_sessionindex, hw_video_t* i_pvideo)
    {
        _sessionindex = i_sessionindex;
        _pvideo = i_pvideo;
    }
    virtual ~CameraDeviceSessionImpl() = default;
    /*
    * Call it when the session instance is not nullptr.
    */
    void ReInit(u32 i_sessionindex, hw_video_t* i_pvideo)
    {
        _sessionindex = i_sessionindex;
        _pvideo = i_pvideo;
        _vthreadpipelinenotif.clear();
        _vthreadblocknotif.clear();
        _vthreadoutput.clear();
        u32 blocki;
        u32 sensori;
        for (blocki = 0; blocki < HW_VIDEO_NUMBLOCKS_MAX; blocki++)
        {
            for (sensori = 0; sensori < HW_VIDEO_NUMSENSORS_PER_BLOCK; sensori++)
            {
                _parray_datacbnum[blocki][sensori] = 0;
            }
        }
    }
protected:
    s32 Internal_RegisterDefaultThreadRoutine();
    /*
    * We prepare the hw_video_blockspipelineconfig_t structure to pipeline_open.
    * We prepare the thread routine.
    * Before call the function, we already set the default thread routine.
    * Currently, _numblocks, _numsensors, _blocktype, _capturewidth, _captureheight, _width, _height
    * are all set in the following function.
    * In later version, some of them may be set dynamically when pipeline is running.
    */
    s32 Internal_PrepareLowerLevelConfig_AndHalBasicInfo();
    /*
    * Add it because the constructor cannot return error.
    * The callback operation should run after the instance is created.
    */
    s32 Internal_Init(ICameraDeviceCallback* i_pcallback);

    static void processFrame(int blockidx,int sensoridx,CameraDeviceSessionImpl* context,camera_device_datacb datacb);
    int         getShmKeyBySensorID(int blockidx,int sensoridx);
public:
    virtual s32 Close();
protected:
    // session index of the session pointer array in CameraDeviceImpl instance
    u32                                                 _sessionindex;
    CAMERA_DEVICE_OPENTYPE                              _opentype;
    CAMERA_DEVICE_OPENMODE                              _openmode;
    // used by CameraDeviceSessionImpl::GetBlockType, never be CAMERA_DEVICE_BLOCK_TYPE_CURRENT_SINGLE_GROUP
    CAMERA_DEVICE_BLOCK_TYPE                            _blocktype;
    // lower level
    u32                                                 _numblocks;
    // lower level, valid when _numblocks is 1
    u32                                                 _numsensors;
    // Currently only consider one sensor type situation.
    u32                                                 _capturewidth;
    // Currently only consider one sensor type situation.
    u32                                                 _captureheight;
    // Currently only consider one sensor type situation. Currently cannot change the origin capture width.
    u32                                                 _width;
    // Currently only consider one sensor type situation. Currently cannot change the origin capture height .
    u32                                                 _height;
    struct hw_video_t*                                  _pvideo;
    camera_device_custom_threadroutine_handleframe_impl     _threadroutine_handleframe_default;
    camera_device_custom_threadroutine_sensornotif_impl     _threadroutine_sensornotif_default;
    camera_device_custom_threadroutine_blocknotif_impl      _threadroutine_blocknotif_default;
    /*
    * The specific sensor different outputtype, set the same handle frame thread routine.
    * When you need to write your custom thread routine, you need to know the lower level concept outputtype.
    */
    camera_device_custom_threadroutine_handleframe_impl     _parray_threadroutine_handleframe[HW_VIDEO_NUMBLOCKS_MAX][HW_VIDEO_NUMSENSORS_PER_BLOCK];
    void*                                               _parray_threadpcontext_handleframe[HW_VIDEO_NUMBLOCKS_MAX][HW_VIDEO_NUMSENSORS_PER_BLOCK];
    camera_device_custom_threadroutine_sensornotif_impl     _parray_threadroutine_sensornotif[HW_VIDEO_NUMBLOCKS_MAX][HW_VIDEO_NUMSENSORS_PER_BLOCK];
    void*                                               _parray_threadpcontext_sensornotif[HW_VIDEO_NUMBLOCKS_MAX][HW_VIDEO_NUMSENSORS_PER_BLOCK];
    camera_device_custom_threadroutine_blocknotif_impl      _parray_threadroutine_blocknotif[HW_VIDEO_NUMBLOCKS_MAX];
    void*                                               _parray_threadpcontext_blocknotif[HW_VIDEO_NUMBLOCKS_MAX];
    struct hw_video_blockspipelineconfig_t              _pipelineconfig;
    hw_video_handlepipeline                             _handlepipeline;
    struct hw_video_blockspipeline_ops_t*               _pblockspipeline_ops;
    // need clear when ReInit
    std::vector<std::unique_ptr<std::thread>>           _vthreadpipelinenotif;
    std::vector<std::unique_ptr<std::thread>>           _vthreadblocknotif;
    std::vector<std::unique_ptr<std::thread>>           _vthreadoutput;
    /*
    * One sensor correspondent to one event callback function.
    */
    camera_device_eventcb                               _parray_eventcb[HW_VIDEO_NUMBLOCKS_MAX][HW_VIDEO_NUMSENSORS_PER_BLOCK] = { nullptr };
    u32                                                 _parray_datacbnum[HW_VIDEO_NUMBLOCKS_MAX][HW_VIDEO_NUMSENSORS_PER_BLOCK] = { 0 };
    CameraDeviceDataCbEnv                               _parray_datacbenv[HW_VIDEO_NUMBLOCKS_MAX][HW_VIDEO_NUMSENSORS_PER_BLOCK][CAMERA_DEVICE_DATACB_TYPE_MAXCOUNT];
    int                                                 _consumer_ser_fd[4][4];
    std::vector<std::thread>                            _handleFramethreads;
    bool                                                _enable_shm = false;
};

#endif
