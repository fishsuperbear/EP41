
#include "hal_camera.hpp"
/* #include "cuda_runtime_api.h" */
/* #include "cuda.h" */
#include <unistd.h>
#include "hw_platform.h"


using namespace std;

unique_ptr<ICameraDevice> _pcameradevice;

CAMERA_DEVICE_OPENTYPE _opentype = CAMERA_DEVICE_OPENTYPE_MULTIROUP_SENSOR_DESAY;


void handle_enc_buffer(CameraDeviceDataCbInfo* i_pbufferinfo)
{
    printf("do handle_encode_buffer++++++++++buffer+ bufsize = %d\r\n", i_pbufferinfo->size);
}
void handle_cuda_buffer(struct CameraDeviceGpuDataCbInfo* i_pbufferinfo)
{
    /* printf("do handle_cuda_buffer++++++++++buffer+ blocktype[%c],sensor[%d]capturetsc[%llu] bufsize = %dpbuffer=%p\r\n", */
    /*     'A' + i_pbufferinfo->blocktype - 1, i_pbufferinfo->sensorindex, i_pbufferinfo->timeinfo.timestamp, */
    /*     i_pbufferinfo->pgpuinfo->prgbinfo->buffsize,i_pbufferinfo->pgpuinfo->prgbinfo->pbuff); */
    u64 tsc;
    hw_plat_get_tsc_ns(&tsc);
    printf(
        "do handle_cuda_buffer++++++++++buffer+ blocktype[%c],sensor[%d]capturetsc[%llu] bufsize = %d, pbuff[%p], "
        "width[%u], height[%u], timestamp[%llu]current[%llu]\r\n",
        'A' + i_pbufferinfo->blocktype - 1, i_pbufferinfo->sensorindex, i_pbufferinfo->timeinfo.timestamp,
        i_pbufferinfo->pgpuinfo->prgbinfo->buffsize, i_pbufferinfo->pgpuinfo->prgbinfo->pbuff, i_pbufferinfo->width,
        i_pbufferinfo->height, i_pbufferinfo->timeinfo.timestamp,tsc);
}
void handle_event(CameraDeviceEventCbInfo* i_peventcbinfo)
{
    // currently do nothing
}

class CameraDeviceCallbackOrin : public ICameraDeviceCallback
{
public:
    virtual CAMERA_DEVICE_OPENTYPE RegisterOpenType()
    {
        return _opentype;
    }
    virtual CAMERA_DEVICE_OPENMODE RegisterOpenMode()
    {
        return CAMERA_DEVICE_OPENMODE_SUB;
    }
    virtual s32 RegisterCallback()
    {
        s32 ret = 0;
        /*
        * Register event callback.
        */
        CameraDeviceEventCbRegInfo eventreginfo;
        eventreginfo.opentype = _opentype;
        // see blocktype note of CameraDeviceDataCbRegInfo
        eventreginfo.blocktype = CAMERA_DEVICE_BLOCK_TYPE_GROUPA;
        // currently only one sensor type in one block, need to change it in the future
        eventreginfo.sensorindex = 0;
        if ((ret = _pcameradevice->RegisterEventCallback(&eventreginfo, handle_event)) < 0) {
            printf("RegisterEventCallback fail! ret=%x\r\n", ret);
        }
        /*
        * Register data callback and gpu data callback.
        */
        CameraDeviceGpuDataCbRegInfo gpudatareginfo;
        u32 sensorindex;
        gpudatareginfo.opentype = _opentype;
        for (int blockidx = 0;blockidx<3;blockidx++){
            switch(blockidx)
            {
                case 0:
                    gpudatareginfo.blocktype =CAMERA_DEVICE_BLOCK_TYPE_GROUPA;
                    break;
                case 1:
                    gpudatareginfo.blocktype =CAMERA_DEVICE_BLOCK_TYPE_GROUPB;
                    break;
                case 2:
                    gpudatareginfo.blocktype =CAMERA_DEVICE_BLOCK_TYPE_GROUPC;
                    break;

            }
            for (sensorindex = 0; sensorindex < 4; sensorindex++)
            {
                gpudatareginfo.sensorindex = sensorindex;
                gpudatareginfo.gpuimgtype = CAMERA_DEVICE_GPUDATACB_IMGTYPE_RGB888_NHWC_BGR;
                gpudatareginfo.interpolation = CAMERA_DEVICE_GPUDATACB_INTERPOLATION_NEAREST;
                gpudatareginfo.busecaptureresolution = 1;
                gpudatareginfo.busecaptureframerate = 1;
                gpudatareginfo.rotatedegrees = 0;
                gpudatareginfo.pcontext = nullptr;
                if ((ret = _pcameradevice->RegisterGpuDataCallback(&gpudatareginfo, handle_cuda_buffer)) < 0) {
                    printf("RegisterDataCallback fail! ret=%x\r\n", ret);
                }
            }
        }

        return 0;
    }
    virtual s32 RegisterCustomThreadRoutine()
    {
        return 0;
    }
};


int main(int argc, char* argv[])
{
    s32 ret = 0;
    CameraDeviceCallbackOrin callbackorin;
    ICameraDeviceSession* psession;
    _pcameradevice = ICameraDevice::GetInstance(HAL_CAMERA_VERSION_1_0);
    if ((ret = _pcameradevice->CreateCameraSession(&callbackorin, &psession)) < 0) {
        printf("Camera Device Open fail! ret=%x\r\n", ret);
        return -1;
    }
    printf("sleep 5 seconds\r\n");
    sleep(100);
    psession->Close();
    return 0;
}
