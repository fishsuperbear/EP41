// Copyright (c) 2022 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from NVIDIA Corporation is strictly prohibited.

#ifndef CCOMMONCCONSUMER_HPP
#define CCOMMONCCONSUMER_HPP

#include "CConsumer.hpp"
#include "NvSIPLClient.hpp"
#include "NvSIPLDeviceBlockInfo.hpp"

typedef uint32_t        u32;

#define HW_NVMEDIA_COMMON_CONSUMER_TESTRAWOUTPUT            0

class CCommonConsumer: public CConsumer
{
public:
    CCommonConsumer() = delete;
    CCommonConsumer(NvSciStreamBlock handle,
        uint32_t uSensor,
        NvSciStreamBlock queueHandle);
    virtual ~CCommonConsumer(void);
protected:
    virtual SIPLStatus HandleClientInit(void) override;
    virtual SIPLStatus SetDataBufAttrList(void) override;
    virtual SIPLStatus SetSyncAttrList(void) override;
    virtual SIPLStatus MapDataBuffer(uint32_t packetIndex) override;
    virtual SIPLStatus RegisterSignalSyncObj(void) override;
    virtual SIPLStatus RegisterWaiterSyncObj(uint32_t index) override;
    virtual SIPLStatus InsertPrefence(uint32_t packetIndex, NvSciSyncFence &prefence) override;
    virtual SIPLStatus SetEofSyncObj(void) override;
    virtual SIPLStatus ProcessPayload(uint32_t packetIndex, NvSciSyncFence *pPostfence) override;
    virtual SIPLStatus UnregisterSyncObjs(void) override;
    virtual SIPLStatus OnProcessPayloadDone(uint32_t packetIndex) override;
    virtual bool HasCpuWait(void) {return true;};

private:
    SIPLStatus InitCommon(NvSciBufAttrList bufAttrList);
    SIPLStatus HandleOneFrame(NvSciBufObj i_scibufobj,
        uint8_t** o_ppoutputbuf, size_t* o_pnumbytes, NvSciSyncFence* o_ppostfence);
    SIPLStatus SetCommonConfig(void);

    NvSciBufObj m_pSciBufObjs[MAX_NUM_PACKETS] {nullptr};
    FILE *m_pOutputFile = nullptr;
    uint16_t m_encodeWidth;
    uint16_t m_encodeHeight;
    uint8_t *m_pbufcpu = nullptr;
    size_t m_encodedBytes = 0;

    typedef struct {
        float heightFactor[MAX_NUM_SURFACES];
        float widthFactor[MAX_NUM_SURFACES];
        uint32_t numSurfaces;
    } BufUtilSurfParams;

    BufUtilSurfParams _bufsurfparamstable_default = {
        .heightFactor = {1, 0, 0},
        .widthFactor = {1, 0, 0},
        .numSurfaces = 1,
    };
    BufUtilSurfParams _bufsurfparamstable_yuv[3] = {
        /* Shift factors for SEMI_PLANAR to PLANAR conversion */
        { /* 420 */
            .heightFactor = {1, 0.5, 0.5},
            .widthFactor = {1, 0.5, 0.5},
            .numSurfaces = 3,
        },
        { /* 444 */
            .heightFactor = {1, 1, 1},
            .widthFactor = {1, 1, 1},
            .numSurfaces = 3,
        },
        { /* 422 */
            .heightFactor = {1, 1, 1},
            .widthFactor = {1, 0.5, 0.5},
            .numSurfaces = 3,
        },
    };

private:
    SIPLStatus checkinittestfile(NvSciBufObj i_scibufobj);
    SIPLStatus getbuffparams(BufferAttrs buffAttrs,
        float** o_pxscale,
        float** o_pyscale,
        uint32_t** o_ppbytesperpixel,
        uint32_t* o_pnumsurfacesval,
        bool* o_pispackedyuv);
    bool getbpp(uint32_t buffBits, uint32_t* buffBytesVal);

private:
    /*
    * Test file macros:     CCOMMONCONSUMER_TESTFILE_FRAMECOUNT
    * Test file variables:  _pfiletest,_testframecount
    * Test file functions:  checkinittestfile
    */
    FILE*               _pfiletest = nullptr;
#if (HW_NVMEDIA_COMMON_CONSUMER_TESTRAWOUTPUT == 1)
    FILE*               _pfiletestraw = nullptr;
#endif
    u32                 _testframecount = 0;
    u32                 _blockindex = 0;
    u32                 _sensorindex = 0;
    INvSIPLClient::ConsumerDesc::OutputType     _outputtype;
    // may delete later
    uint8_t*            m_pBuff = nullptr;
};

#endif
