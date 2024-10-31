#ifndef ICAMERADEVICE_HPP
#define ICAMERADEVICE_HPP

#include "camera_types.hpp"
#include "ICameraDeviceCallback.hpp"
#include "ICameraDeviceSession.hpp"

class ICameraDevice
{
    /*
     * You properly use the following interfaces.
     */
    public:
        /*
         * One process can own one instance at one time.
         * return not null means valid.
         * When the instance release it will ensure all of the current device has closed and release all of the
         * resource of the current process.
         */
        static std::unique_ptr<ICameraDevice> GetInstance(HAL_CAMERA_VERSION i_version);
        /*
         * CreateCameraSession camera device according to the camera open type.
         * The function will open device and start device pipeline immediately.
         * You do not need to call other pipeline start function.
         * *o_ppsession output the session pointer interpret the session.
         * Use ICameraDeviceSession::Close function to stop the pipeline and close the device.
         */
        virtual s32 CreateCameraSession(ICameraDeviceCallback* i_pcallback, ICameraDeviceSession** o_ppsession) = 0;
        /*
         * You can only call it in your implemented ICameraDeviceCallback::RegisterEventCallback function.
         * The input i_eventcb should always be valid though the whole camera device open->close life cycle.
         */
        virtual s32 RegisterEventCallback(CameraDeviceEventCbRegInfo* i_peventreginfo, camera_device_eventcb i_eventcb) = 0;
        /*
         * Not gpu data.
         * You can only call it in your implemented ICameraDeviceCallback::RegisteDataCb function.
         * The input i_datacb should always be valid though the whole camera device open->close life cycle.
         */
        virtual s32 RegisterCpuDataCallback(const CameraDeviceCpuDataCbRegInfo* i_pdatareginfo, camera_device_datacb i_datacb) = 0;
        /*
         * Gpu data.
         * You can only call it in your implemented ICameraDeviceCallback::RegisteDataCb function.
         * The input i_gpudatacb should always be valid though the whole camera device open->close life cycle.
         */
        virtual s32 RegisterGpuDataCallback(const CameraDeviceGpuDataCbRegInfo* i_pgpureginfo, camera_device_gpudatacb i_gpudatacb) = 0;

    public:
        /*
         * You should delete it by yourself when you do not use the ICameraDevice instance.
         */
        virtual ~ICameraDevice() = default;
};

#endif
