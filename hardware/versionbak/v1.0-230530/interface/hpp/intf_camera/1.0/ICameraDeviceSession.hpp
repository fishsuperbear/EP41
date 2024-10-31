#ifndef ICAMERADEVICESESSION_HPP
#define ICAMERADEVICESESSION_HPP

#include "camera_types.hpp"

/*
 * You do not need to release it by yourself.
 * The interface is invalid after you called ICameraDeviceSession::Close().
 * Another situation that the interface is invalid is after you release the ICameraDevice
 * instance in abnormal cases.
 */
class ICameraDeviceSession
{
    /*
     * You properly use the following interfaces.
     */
    public:
        /*
         * Stop the pipeline and close the device.
         */
        virtual s32 Close() = 0;

    protected:
        /*
         * You cannot new it by yourself.
         */
        ICameraDeviceSession() = default;
        /*
         * You cannnot delete it by yourself.
         */
        virtual ~ICameraDeviceSession() = default;
};

#endif
