#pragma once

#include "nvscistream.h"
#include "sensor/nvs_adapter/nvs_logger.h"
#include "sensor/nvs_adapter/nvs_helper.h"
#include <string>
#include <thread>

namespace hozon {
namespace netaos {
namespace nv { 

#define MAX_ELEMS 4
#define MAX_PACKETS 32
#define INFO_SIZE 50
#define MAX_CONSUMERS 4

struct ElemAttr {
    /* The application's name for the element */
    uint32_t userName;
    /* Attribute list for element */
    NvSciBufAttrList attrList;
};

#define ENDINFO_NAME_PROC 0xabcd
#define ELEMENT_NAME_IMAGE 0xbeef
#define ELEMENT_NAME_METADATA 0xaaaa

class NVSBlockCommon {
public:
    int32_t EventHandler();
    virtual void Stop();

    std::string name;
    NvSciStreamBlock block;
    
protected:
    virtual void DeleteBlock();
    virtual int32_t OnError();
    virtual int32_t OnDisconnected();
    virtual int32_t OnConnected();
    virtual int32_t OnSetupComplete();
    virtual int32_t OnElements();
    virtual int32_t OnPacketCreate();
    virtual int32_t OnPacketsComplete();
    virtual int32_t OnWaiterAttr();
    virtual int32_t OnSignalObj();
    virtual int32_t OnPacketReady();
    virtual int32_t OnPacketStatus();

    virtual void RegIntoEventService();
    virtual void DeleteEventService();
    
    uint32_t _wait_us = 1e6;
    std::shared_ptr<std::thread> _event_thread;

private:
    void Loop();

    bool _running = true;
};

template<typename T>
T DumpBufAttr(NvSciBufAttrList& attr, NvSciBufAttrKey key, const std::string& name) {
    T ret = 0;

    NvSciBufAttrKeyValuePair keyVals[] = {
        {key, NULL, 0},
    };

    auto err = NvSciBufAttrListGetAttrs(attr, keyVals, 1);
    if (err != NvSciError_Success) {
        NVS_LOG_DEBUG << "****Attr " << name << ": null";
        return ret;
    }

    if (keyVals[0].value == nullptr) {
        NVS_LOG_DEBUG << "****Attr " << name << ": empty";
        return ret;
    }
    else {
        NVS_LOG_DEBUG << "****Attr " << name << ": " << *(T *)(keyVals[0].value);
        return *(T *)(keyVals[0].value);
    }
}

template<typename T>
void DumpBufAttr(NvSciBufAttrList& attr, NvSciBufAttrKey key, const std::string& name, uint32_t size) {
    NvSciBufAttrKeyValuePair keyVals[] = {
        {key, NULL, 0},
    };

    auto err = NvSciBufAttrListGetAttrs(attr, keyVals, 1);
    if (err != NvSciError_Success) {
        NVS_LOG_DEBUG << "****Attr " << name << ": null";
        return;
    }

    if (keyVals[0].value == nullptr) {
        NVS_LOG_DEBUG << "****Attr " << name << ": empty";
        return;
    }
    else {
        std::string out;
        for (uint32_t i = 0; i < size; ++i) {
            out += std::to_string(*((T *)(keyVals[0].value) + i)) + " ";
        }

        NVS_LOG_DEBUG << "****Attr " << name << ": { " << out << "}";
    }
}

void DumpBufAttrAll(NvSciBufAttrList& attr);
void DumpBufAttrAll(const std::string& title, NvSciBufAttrList& attr);
void DumpBufAttrAll(NvSciBufObj& obj);

}
}
}

#define NVS_ASSERT_RETURN_INT(statement) \
    if ((statement) < 0) { \
        return -1; \
    }

#define NVS_ASSERT_RETURN_STREAM(statement, desp) \
    stream_err = (statement); \
    if (stream_err != NvSciError_Success) { \
        NVS_LOG_CRITICAL << "Fail to " << desp << ", ret " << LogHexNvErr(stream_err); \
        return -1; \
    }

#define NVS_ASSERT_RETURN_MEDIA(statement, desp) \
    media_err = (statement); \
    if (media_err != NVMEDIA_STATUS_OK ) { \
        NVS_LOG_CRITICAL << "Fail to " << desp << ", ret " << media_err; \
        return -1; \
    }

#define NVS_ASSERT_RETURN_CUDA(statement, desp) \
    cuda_err = (statement); \
    if (cuda_err != CUDA_SUCCESS ) { \
        NVS_LOG_CRITICAL << "Fail to " << desp << ", ret " << log::loghex((uint32_t)cuda_err); \
        return -1; \
    }
