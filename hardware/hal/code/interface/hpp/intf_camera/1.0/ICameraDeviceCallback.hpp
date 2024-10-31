#ifndef ICAMERADEVICECALLBACK_HPP
#define ICAMERADEVICECALLBACK_HPP

#include "camera_types.hpp"

class ICameraDeviceCallback
{
    public:
        /*
         * You should return the correct device open type.
         * Called only when ICameraDevice::CreateCameraSession function.
         * Will not use it again after ICameraDevice::CreateCameraSession.
         */
        virtual CAMERA_DEVICE_OPENTYPE RegisterOpenType() = 0;
        /*
         * You should return the correct device open mode.
         * Called only when ICameraDevice::CreateCameraSession function.
         * Will not use it again after ICameraDevice::CreateCameraSession.
         */
        virtual CAMERA_DEVICE_OPENMODE RegisterOpenMode() = 0;
        /*
         * You should call ICameraDevice::RegisterEventCallback to register event callback.
         * You should call ICameraDevice::RegisterDataCallback to register the interested data
         * callback of the input type (not gpu).
         * You should call ICameraDevice::RegisterGpuDataCallback to register the interested data
         * callback of the input type (gpu).
         * Called only when ICameraDevice::CreateCameraSession function.
         * Will not use it again after ICameraDevice::CreateCameraSession.
         */
        virtual s32 RegisterCallback() = 0;
};

#endif
