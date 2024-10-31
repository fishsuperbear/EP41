#include "hal_camera.hpp"
#include <unistd.h>

using namespace std;

unique_ptr<ICameraDevice> _pcameradevice;

CAMERA_DEVICE_OPENTYPE _opentype = CAMERA_DEVICE_OPENTYPE_MULTIROUP_SENSOR_DESAY_SUB;

void onFrameAvailable(CameraDeviceDataCbInfo* i_pbufferinfo)
{
    printf("onFrameAvailable++ get blocktype[%c],sensor[%d]capturetsc[%llu]buffersize=%d\n", 'A' + i_pbufferinfo->blocktype - 1, i_pbufferinfo->sensorindex, i_pbufferinfo->timeinfo.framecapturetsc, i_pbufferinfo->size);
    // 处理帧数据
}

class CameraDeviceCallbackOrin : public ICameraDeviceCallback {
public:
    virtual CAMERA_DEVICE_OPENTYPE RegisterOpenType() {
        return _opentype;
    }
    virtual CAMERA_DEVICE_OPENMODE RegisterOpenMode() {
        return CAMERA_DEVICE_OPENMODE_ENC_SHM_SUB;
    }
    virtual s32 RegisterCallback() {
        s32 ret = 0;
        /*
		* Register data callback and gpu data callback.
		*/
        CameraDeviceDataCbRegInfo datareginfo;
        u32 sensorindex;
        datareginfo.opentype = _opentype;
        for (int blockidx = 0; blockidx < 3; blockidx++) {
            switch (blockidx) {
                case 0:
                    datareginfo.blocktype = CAMERA_DEVICE_BLOCK_TYPE_GROUPA;
                    break;
                case 1:
                    datareginfo.blocktype = CAMERA_DEVICE_BLOCK_TYPE_GROUPB;
                    break;
                case 2:
                    datareginfo.blocktype = CAMERA_DEVICE_BLOCK_TYPE_GROUPC;
                    break;
            }
            for (sensorindex = 0; sensorindex < 4; sensorindex++) {
                if (blockidx == 0 && sensorindex == 3) {
                    continue;
                }
                printf("RegisterCallback\n");
                // see blocktype note of CameraDeviceDataCbRegInfo
                // currently only one sensor type in one block, need to change it in the future
                datareginfo.sensortype = CAMERA_DEVICE_SENSOR_TYPE_CURRENT_SINGLE_SENSOR;
                datareginfo.sensorindex = sensorindex;
                datareginfo.datacbtype = CAMERA_DEVICE_DATACB_TYPE_HEVC;
                datareginfo.busecaptureresolution = 1;
                // set it to 0 currently
                datareginfo.rotatedegrees = 0;
                // you can set your custom context pointer
                datareginfo.pcontext = nullptr;
                if ((ret = _pcameradevice->RegisterDataCallback(&datareginfo, onFrameAvailable)) < 0) {
                    printf("RegisterDataCallback fail! ret=%x\r\n", ret);
                }
            }
        }

        return 0;
    }
    virtual s32 RegisterCustomThreadRoutine() {
        return 0;
    }
};

int main(int argc, char* argv[]) {
    s32 ret = 0;
    CameraDeviceCallbackOrin callbackorin;
    ICameraDeviceSession* psession;
    _pcameradevice = ICameraDevice::GetInstance(HAL_CAMERA_VERSION_0_1, false);
    if ((ret = _pcameradevice->CreateCameraSession(&callbackorin, &psession)) < 0) {
        printf("Camera Device Open fail! ret=%x\r\n", ret);
        return -1;
    }
    printf("sleep 5 seconds\r\n");
    sleep(500);
    psession->Close();
    return 0;
}
