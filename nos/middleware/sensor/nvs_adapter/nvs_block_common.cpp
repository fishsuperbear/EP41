#include "sensor/nvs_adapter/nvs_block_common.h"
#include "sensor/nvs_adapter/nvs_logger.h"

namespace hozon {
namespace netaos {
namespace nv { 

void NVSBlockCommon::RegIntoEventService() {
    _running = true;
    _event_thread = std::make_shared<std::thread>(&NVSBlockCommon::Loop, this);
}

void NVSBlockCommon::DeleteEventService() {
    _running = false;
    _event_thread->join();
}

int32_t NVSBlockCommon::EventHandler() {
    NvSciStreamEventType event;
    NvSciError err;

    _wait_us = 300 * 1000;
    // NVS_LOG_DEBUG << "block " << block << ", _wait_us " << _wait_us << ", event " << log::loghex((uint32_t)event);
    err = NvSciStreamBlockEventQuery(block, _wait_us, &event);
    if (NvSciError_Success != err) {
        if (NvSciError_Timeout == err) {
            // NVS_LOG_WARN << name << " timed out waiting for setup instructions";
            return 0;
        } 
        else {
            NVS_LOG_ERROR << name << " event query failed with error " << LogHexNvErr(err);
        }

        // DeleteBlock();

        return -1;
    }

    /* If we received an event, handle it based on its type */
    int32_t rv = 1;
    // NvSciError status;
    switch (event) {
    default:
        NVS_LOG_WARN << name << " recv unknown event " << log::loghex((uint32_t)event);
        rv = -1;
        break;

    /*
     * Error events should never occur with safety-certified drivers,
     *   and are provided only in non-safety builds for debugging
     *   purposes. Even then, they should only occur when something
     *   fundamental goes wrong, like the system running out of memory,
     *   or stack/heap corruption, or a bug in NvSci which should be
     *   reported to NVIDIA.
     */
    case NvSciStreamEventType_Error:
        NVS_LOG_DEBUG << name << " recv NvSciStreamEventType_Error";
        OnError();
        rv = -1;
        break;

    /*
     * If told to disconnect, it means either the stream finished its
     *   business or some other block had a failure. We'll just do a
     *   clean up and return without an error.
     */
    case NvSciStreamEventType_Disconnected:
        NVS_LOG_DEBUG << name << " recv NvSciStreamEventType_Disconnected";
        OnDisconnected();
        rv = 2;
        break;

    /*
     * The block doesn't have to do anything on connection, but now we may
     *   wait forever for any further events, so the timeout becomes infinite.
     */
    case NvSciStreamEventType_Connected:
        NVS_LOG_DEBUG << name << " recv NvSciStreamEventType_Connected";
        if (OnConnected() < 0) {
            rv = -1;
        }
        break;

    /* All setup complete. Transition to runtime phase */
    case NvSciStreamEventType_SetupComplete:
        NVS_LOG_DEBUG << name << " recv NvSciStreamEventType_SetupComplete";
        if (OnSetupComplete() < 0) {
            rv = -1;
        }
        break;

    /* Retrieve all element information from pool */
    case NvSciStreamEventType_Elements:
        NVS_LOG_DEBUG << name << " recv NvSciStreamEventType_Elements";
        if (OnElements() < 0) {
            rv = -1;
        }
        break;

    /* For a packet, set up an entry in the array */
    case NvSciStreamEventType_PacketCreate:
        NVS_LOG_DEBUG << name << " recv NvSciStreamEventType_PacketCreate";
        if (OnPacketCreate() < 0) {
            rv = -1;
        }
        break;

    /* Finish any setup related to packet resources */
    case NvSciStreamEventType_PacketsComplete:
        NVS_LOG_DEBUG << name << " recv NvSciStreamEventType_PacketsComplete";
        if (OnPacketsComplete() < 0) {
            rv = -1;
        }
        break;

        /* Set up signaling sync object from consumer's wait attributes */
    case NvSciStreamEventType_WaiterAttr:
        NVS_LOG_DEBUG << name << " recv NvSciStreamEventType_WaiterAttr";
        if (OnWaiterAttr() < 0) {
            rv = -1;
        }
        break;

    /* Import producer sync objects for all elements */
    case NvSciStreamEventType_SignalObj:
        NVS_LOG_DEBUG << name << " recv NvSciStreamEventType_SignalObj";
        if (OnSignalObj() < 0) {
            rv = -1;
        }
        break;

    /* Processs payloads when packets arrive */
    case NvSciStreamEventType_PacketReady:
        NVS_LOG_DEBUG << name << " recv NvSciStreamEventType_PacketReady";
        if (OnPacketReady() < 0) {
            rv = -1;
        }
        break;

    case NvSciStreamEventType_PacketStatus:
        NVS_LOG_DEBUG << name << " recv NvSciStreamEventType_PacketStatus";
        if (OnPacketStatus() < 0) {
            rv = -1;
        }
        break;
    }

    // /* On failure or final event, clean up the block */
    // if ((rv < 0) || (1 < rv)) {
    //     DeleteBlock();
    // }

    return rv;
}

void NVSBlockCommon::Stop() {
    DeleteEventService();
    DeleteBlock();
}

void NVSBlockCommon::DeleteBlock() {
    NvSciStreamBlockDelete(block);
}

int32_t NVSBlockCommon::OnError() {
    NvSciError status;
    NvSciError err;

    err = NvSciStreamBlockErrorGet(block, &status);
    if (NvSciError_Success != err) {
        NVS_LOG_ERROR << name << " failed to query the error event code " << LogHexNvErr(err);
    } 
    else {
        NVS_LOG_ERROR << name << " received error event: " << LogHexNvErr(status);
    }

    return 0;
}

int32_t NVSBlockCommon::OnDisconnected() {
    return 0;
}

int32_t NVSBlockCommon::OnConnected() {
    return 0;
}

int32_t NVSBlockCommon::OnSetupComplete() {
    return 0;
}

int32_t NVSBlockCommon::OnElements() {
    return 0;
}

int32_t NVSBlockCommon::OnPacketCreate() {
    return 0;
}

int32_t NVSBlockCommon::OnPacketsComplete() {
    return 0;
}

int32_t NVSBlockCommon::OnWaiterAttr() {
    return 0;
}

int32_t NVSBlockCommon::OnSignalObj() {
    return 0;
}

int32_t NVSBlockCommon::OnPacketReady() {
    return 0;
}

int32_t NVSBlockCommon::OnPacketStatus() {
    return 0;
}

void NVSBlockCommon::Loop() {
    int32_t ret = 0;

    while (_running) {
        ret = EventHandler();
        if (ret < 0) {
            return;
        }
    }
}

void DumpBufAttrAll(NvSciBufAttrList& attr) {
    // DumpBufAttr<int32_t>(attr, NvSciBufGeneralAttrKey_RequiredPerm, "NvSciBufGeneralAttrKey_RequiredPerm");
    DumpBufAttr<int32_t>(attr, NvSciBufGeneralAttrKey_Types, "NvSciBufGeneralAttrKey_Types");
    uint32_t plane_cnt = DumpBufAttr<uint32_t>(attr, NvSciBufImageAttrKey_PlaneCount, "NvSciBufImageAttrKey_PlaneCount");
    DumpBufAttr<int32_t>(attr, NvSciBufImageAttrKey_PlaneColorFormat, "NvSciBufImageAttrKey_PlaneColorFormat", plane_cnt);
    DumpBufAttr<int32_t>(attr, NvSciBufImageAttrKey_Layout, "NvSciBufAttrValImageLayoutType");
    DumpBufAttr<uint64_t>(attr, NvSciBufImageAttrKey_Size, "NvSciBufImageAttrKey_Size");
    DumpBufAttr<uint64_t>(attr, NvSciBufImageAttrKey_Alignment, "NvSciBufImageAttrKey_Alignment");
    DumpBufAttr<uint32_t>(attr, NvSciBufImageAttrKey_PlaneWidth, "NvSciBufImageAttrKey_PlaneWidth", plane_cnt);
    DumpBufAttr<uint32_t>(attr, NvSciBufImageAttrKey_PlaneHeight, "NvSciBufImageAttrKey_PlaneHeight", plane_cnt);
    DumpBufAttr<uint64_t>(attr, NvSciBufImageAttrKey_PlaneAlignedSize, "NvSciBufImageAttrKey_PlaneAlignedSize", plane_cnt);
    DumpBufAttr<uint32_t>(attr, NvSciBufImageAttrKey_PlanePitch, "NvSciBufImageAttrKey_PlanePitch", plane_cnt);
    DumpBufAttr<uint32_t>(attr, NvSciBufImageAttrKey_PlaneBitsPerPixel, "NvSciBufImageAttrKey_PlaneBitsPerPixel", plane_cnt);
    DumpBufAttr<uint32_t>(attr, NvSciBufImageAttrKey_PlaneAlignedHeight, "NvSciBufImageAttrKey_PlaneAlignedHeight", plane_cnt);
    DumpBufAttr<uint32_t>(attr, NvSciBufImageAttrKey_PlaneBaseAddrAlign, "NvSciBufImageAttrKey_PlaneBaseAddrAlign", plane_cnt);
    DumpBufAttr<bool>(attr, NvSciBufGeneralAttrKey_NeedCpuAccess, "NvSciBufGeneralAttrKey_NeedCpuAccess");
    DumpBufAttr<bool>(attr, NvSciBufGeneralAttrKey_EnableCpuCache, "NvSciBufGeneralAttrKey_EnableCpuCache");
    DumpBufAttr<uint64_t>(attr, NvSciBufImageAttrKey_TopPadding, "NvSciBufImageAttrKey_TopPadding");
    DumpBufAttr<uint64_t>(attr, NvSciBufImageAttrKey_BottomPadding, "NvSciBufImageAttrKey_BottomPadding");
    DumpBufAttr<uint64_t>(attr, NvSciBufImageAttrKey_LeftPadding, "NvSciBufImageAttrKey_LeftPadding");
    DumpBufAttr<uint64_t>(attr, NvSciBufImageAttrKey_RightPadding, "NvSciBufImageAttrKey_RightPadding");
    DumpBufAttr<int32_t>(attr, NvSciBufImageAttrKey_PlaneColorStd, "NvSciBufImageAttrKey_PlaneColorStd");
    DumpBufAttr<int32_t>(attr, NvSciBufImageAttrKey_ScanType, "NvSciBufImageAttrKey_ScanType");
}

void DumpBufAttrAll(const std::string& title, NvSciBufAttrList& attr) {
    NVS_LOG_DEBUG << "--------------- " << title << " ATTR ---------------";
    DumpBufAttrAll(attr);
    NVS_LOG_DEBUG << "-------------------------------------------------";
}

void DumpBufAttrAll(NvSciBufObj& obj) {
    NvSciError stream_err;
    NvSciBufAttrList attr;
    // stream_err = NvSciBufAttrListCreate(NVSHelper::GetInstance().sci_buf_module, &attr);
    // if (stream_err != NvSciError_Success) {
    //     NVS_LOG_ERROR << "Fail to create attr, ret " << LogHexNvErr(stream_err);
    //     return;
    // }

    stream_err = NvSciBufObjGetAttrList(obj, &attr);
    if (stream_err != NvSciError_Success) {
        NVS_LOG_ERROR << "Fail to get attr, ret " << LogHexNvErr(stream_err);
        return;
    }

    DumpBufAttrAll(attr);
}

}
}
}