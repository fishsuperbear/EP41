// Copyright (c) 2022 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from NVIDIA Corporation is strictly prohibited.

#ifndef CCUDACONSUMER_H
#define CCUDACONSUMER_H

#include "sensor/nvs_consumer/CConsumer.hpp"
#include "NvSIPLDeviceBlockInfo.hpp"

// cuda includes
#include "cuda_runtime_api.h"
#include "cuda.h"

#include <functional>
#include <memory>

namespace hozon {
namespace netaos {
namespace desay { 

struct DesayCUDAPacket {
    NvSciSyncFence* post_fence;

    void* cuda_dev_ptr;
    std::string format;

    uint64_t data_size;
    uint64_t width;
    uint64_t height;
    uint64_t step;
    
    uint64_t capture_start_us;

    bool need_user_free;
};

class CCudaConsumer: public CConsumer
{
    public:
        CCudaConsumer() = delete;
        CCudaConsumer(NvSciStreamBlock handle, uint32_t uSensor, NvSciStreamBlock queueHandle);
        virtual ~CCudaConsumer(void);

        static SIPLStatus GetBufAttrList(NvSciBufAttrList outBufAttrList);
        static SIPLStatus GetSyncWaiterAttrList(NvSciSyncAttrList outWaiterAttrList);
        using PacketReadyCallback = std::function<void(std::shared_ptr<DesayCUDAPacket>)>;
        void SetOnPacketCallback(PacketReadyCallback callback);
        void ReleasePacket(bool need_free, void* cuda_dev_ptr);

    protected:
        virtual SIPLStatus HandleClientInit(void) override;
        virtual SIPLStatus SetDataBufAttrList(NvSciBufAttrList &bufAttrList) override;
        virtual SIPLStatus SetSyncAttrList(NvSciSyncAttrList &signalerAttrList, NvSciSyncAttrList &waiterAttrList) override;
        virtual SIPLStatus MapDataBuffer(uint32_t packetIndex, NvSciBufObj bufObj) override;
        virtual SIPLStatus RegisterSignalSyncObj(NvSciSyncObj signalSyncObj) override;
        virtual SIPLStatus RegisterWaiterSyncObj(NvSciSyncObj waiterSyncObj) override;
        virtual SIPLStatus InsertPrefence(uint32_t packetIndex, NvSciSyncFence &prefence) override;
        virtual SIPLStatus ProcessPayload(uint32_t packetIndex, NvSciSyncFence *pPostfence) override;
        virtual SIPLStatus OnProcessPayloadDone(uint32_t packetIndex) override;
        virtual SIPLStatus UnregisterSyncObjs(void) override;
        virtual bool HasCpuWait(void) {return true;};
    private:
        SIPLStatus InitCuda(void);
        SIPLStatus BlToPlConvert(uint32_t packetIndex, void *dstptr);

        int m_cudaDeviceId = 0;
        uint8_t *m_pCudaCopyMem[MAX_NUM_PACKETS];
        void  *m_devPtr[MAX_NUM_PACKETS];
        cudaExternalMemory_t m_extMem[MAX_NUM_PACKETS];
        cudaStream_t m_streamWaiter = nullptr;
        cudaExternalSemaphore_t m_signalerSem;
        cudaExternalSemaphore_t m_waiterSem;
        BufferAttrs m_bufAttrs[MAX_NUM_PACKETS] = {};
        cudaMipmappedArray_t m_mipmapArray[MAX_NUM_PACKETS][MAX_NUM_SURFACES] = {};
        cudaArray_t m_mipLevelArray[MAX_NUM_PACKETS][MAX_NUM_SURFACES] = {};

        FILE *m_pOutputFile = nullptr;
        uint8_t *m_pHostBuf = nullptr;
        size_t m_hostBufLen;
	    bool m_FirstCall;
        PacketReadyCallback _on_packet_cb = nullptr;
        void* m_cuda_dev_pl_ptr[MAX_NUM_PACKETS];
        bool is_buffer_init = false;
        HalBufferManager _halbuffer_manager;
    };

}
}
}

#endif