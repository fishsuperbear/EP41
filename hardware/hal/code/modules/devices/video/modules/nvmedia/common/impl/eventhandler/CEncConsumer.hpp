// Copyright (c) 2022 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from NVIDIA Corporation is strictly prohibited.

#ifndef CENCCONSUMER_HPP
#define CENCCONSUMER_HPP

#include "hw_nvmedia_eventhandler_common_impl.h"
#include "hw_nvmedia_eventhandler_outputpipeline_impl.h"
#include "hw_plat_basedef.h"

#include "CConsumer.hpp"
#include "NvSIPLClient.hpp"
#include "nvmedia_iep.h"
#include "NvSIPLDeviceBlockInfo.hpp"

class CEncConsumer: public CConsumer
{
public:
    CEncConsumer() = delete;
    CEncConsumer(NvSciStreamBlock handle, u32 uSensor,
                        u32 i_blockindex, u32 i_sensorindex,
                        NvSciStreamBlock queueHandle,
                        uint16_t encodeWidth,
                        uint16_t encodeHeight,
                        int encodeType, void *i_pvicconsumer);
    virtual ~CEncConsumer(void);

private:
    void *_pvicconsumer = nullptr;

public:
    virtual hw_ret_s32 RegisterDirectCb(struct hw_video_sensorpipelinedatacbconfig_t* i_pcbconfig,
        HWNvmediaEventHandlerRegDataCbConfig* i_peventhandlercbconfig);
protected:
    virtual SIPLStatus HandleClientInit() override;
    virtual SIPLStatus SetDataBufAttrList(NvSciBufAttrList &bufAttrList) override;
    virtual SIPLStatus SetSyncAttrList(NvSciSyncAttrList &signalerAttrList, NvSciSyncAttrList &waiterAttrList) override;
    virtual SIPLStatus MapDataBuffer(uint32_t packetIndex, NvSciBufObj bufObj) override;
    virtual SIPLStatus RegisterSignalSyncObj(NvSciSyncObj signalSyncObj) override;
    virtual SIPLStatus RegisterWaiterSyncObj(NvSciSyncObj waiterSyncObj) override;
    virtual SIPLStatus InsertPrefence(uint32_t packetIndex, NvSciSyncFence &prefence) override;
    virtual SIPLStatus SetEofSyncObj(void) override;
    virtual SIPLStatus ProcessPayload(uint32_t packetIndex, NvSciSyncFence *pPostfence) override;
    virtual SIPLStatus UnregisterSyncObjs(void) override;
    virtual SIPLStatus OnProcessPayloadDone(uint32_t packetIndex) override;
    virtual bool HasCpuWait(void)
    {
        return true;
    };
    virtual SIPLStatus OnDataBufAttrListReceived(NvSciBufAttrList bufAttrListt) override;

private:
    struct DestroyNvMediaIEP
    {
        void operator ()(NvMediaIEP *p) const
        {
            NvMediaIEPDestroy(p);
        }
    };
    SIPLStatus InitEncoder(NvSciBufAttrList bufAttrList);
    SIPLStatus EncodeOneFrame(NvSciBufObj pSciBufObj,
                                u32 i_packetindex,
                                uint8_t **ppOutputBuffer,
                                size_t *pNumBytes,
                                NvSciSyncFence *pPostfence);
    SIPLStatus SetEncodeConfig(void);
    int        getShmKeyBySensorID(int blockidx,int sensoridx);
    static void ProcessEventHandle(CEncConsumer* context);
    static void ProcessClientEventHandle(CEncConsumer* context,int clientfd);

    std::unique_ptr<NvMediaIEP, DestroyNvMediaIEP> m_pNvMIEP{ nullptr };
    NvSciBufObj m_pSciBufObjs[MAX_NUM_PACKETS]{ nullptr };
    NvSciSyncObj m_IEPSignalSyncObj = nullptr;
    FILE *m_pOutputFile = nullptr;
    NvMediaEncodeConfigH264 m_stEncodeConfigH264Params;
    NvMediaEncodeConfigH265 m_stEncodeConfigH265Params;
    uint16_t m_encodeWidth;
    uint16_t m_encodeHeight;
    int m_encodeType;
    uint8_t *m_pEncodedBuf = nullptr;
    size_t m_encodedBytes = 0;

    u32                 _testframecount = 0;
    u32                 _blockindex = 0;
    u32                 _sensorindex = 0;
    INvSIPLClient::ConsumerDesc::OutputType     _outputtype;
    hw_video_sensorpipelinedatacb   _datacb = nullptr;
    void* _pcontext = nullptr;
    // record it down for bak, the user need data type
    HW_VIDEO_REGDATACB_TYPE         _regdatacbtype;
    // set when do init test file operation
    HW_VIDEO_BUFFERFORMAT_MAINTYPE  _orinmaintype;
    // set when do init test file operation
    HW_VIDEO_BUFFERFORMAT_SUBTYPE   _orinsubtype;
    // set when do register data callback operation
    HW_VIDEO_BUFFERFORMAT_MAINTYPE  _usermaintype;
    // set when do register data callback operation
    HW_VIDEO_BUFFERFORMAT_SUBTYPE   _usersubtype;
    // 0 or 1
    u32                             _bneedgetpixel = 0;
    // the expected origin sub type format
    HW_VIDEO_BUFFERFORMAT_SUBTYPE   _expectedoriginsubtype;
    // 0 or 1, 1 means has log that the _datacb is nullptr tip log
    u32                             _blogdatacbnull = 0;
    // 0 or 1
    u32                             _bsynccb = 1;
public:
    int                             _shmid;
    void*                           _shmaddr = nullptr;
    int                             _enc_server_fd;
    char                            _socket_path[100];
    std::mutex                      _rwlock;
    std::vector<int>                _client;
    int                             _alivecount=0;
    int                             _clientready=0;
    int                             _skipfreamCount=0;
};
#endif
