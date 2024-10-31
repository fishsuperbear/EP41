// Copyright (c) 2022 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from NVIDIA Corporation is strictly prohibited.

#ifndef CVICCCONSUMER_H
#define CVICCCONSUMER_H

#include "hw_nvmedia_eventhandler_common_impl.h"
#include "hw_nvmedia_eventhandler_outputpipeline_impl.h"

#include "CConsumer.hpp"
#include "NvSIPLClient.hpp"
#include "NvSIPLDeviceBlockInfo.hpp"
#include "nvmedia_2d.h"
#include "nvmedia_2d_sci.h"
#include "CQueue.hpp"
#include "CBuffer.hpp"
#include "CBufferPool.hpp"
#include "ICascadedProvider.hpp"


class CCascadedMaster;

class CVICConsumer: public CConsumer, public ICascadedProvider
{
public:
    CVICConsumer( NvSciStreamBlock handle,
                    SensorInfo* pSensorInfo, 
                    NvSciStreamBlock queueHandle,
                HWNvmediaOutputPipelineContext* i_poutputpipeline,
                int encodeType);
    virtual ~CVICConsumer();
    virtual SIPLStatus GetNvSciBufAttrList( NvSciBufAttrList attrList ) override;
    virtual SIPLStatus RegisterNvSciBufObjs( const vector<NvSciBufObj>& outputNvSciBufOjbs ) override;

public:
    u64 GetCurrentFrameCaptureTSC();
    u64 GetCurrentFrameCaptureStartTSC();
    u64 GetCurrentExposureStartTime();
protected:
    /*
    * The value is about GetCurrentFrameCaptureTSC.
    */
    u32 _currentpacketindex = 0;

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

private:
    SIPLStatus FillMedia2DOutputBufAttrList( NvSciBufAttrList bufAttrList );
    SIPLStatus FillEncoderInputBufAttrList( NvSciBufAttrList bufAttrList );
    SIPLStatus AllocateSyncObjects();
    SIPLStatus UnregisterSciBufObjects();

private:
    SIPLStatus HandleVICOutput(NvSciBufObj i_scibufobj, NvSciSyncFence* o_ppostfence);
    hw_ret_s32 checktype_and_inittestfile_vic(NvSciBufObj i_scibufobj, HW_VIDEO_BUFFERFORMAT_SUBTYPE i_expectedoriginsubtype);
    SIPLStatus getbuffparams(BufferAttrs buffAttrs,
        float** o_pxscale,
        float** o_pyscale,
        uint32_t** o_ppbytesperpixel,
        uint32_t* o_pnumsurfacesval,
        bool* o_pispackedyuv);
    bool getbpp(uint32_t buffBits, uint32_t* buffBytesVal);

private:
    struct DestroyNvMedia2D
    {
        void operator()( NvMedia2D* handle ) const
        {
            if ( handle )
            {
                NvMediaStatus result = NvMedia2DDestroy( handle );
                if ( result == NVMEDIA_STATUS_OK )
                {
                    handle = nullptr;
                }
                else
                {
                    LOG_ERR( "NvMedia2DDestroy failed with %d\n", result );
                }
            }
        }
    };

    
    /*
    * 0 or 1
    */
    u32                                     _binittestfile = 0;
    u32                                     _testframecount = 0;
#if (HW_NVMEDIA_COMMON_CONSUMER_BASIC_TESTRAWOUTPUT_VIC == 1)
    /*
    * Test file macros:     CCOMMONCONSUMER_TESTFILE_FRAMECOUNT
    * Test file variables:  _pfiletest,_testframecount
    * Test file functions:  checktype_and_inittestfile
    */
    FILE*                                   _pfiletest = nullptr;
#endif

    SensorInfo*                              m_pSensorInfo;
    uint16_t                                 m_encodeWidth;
    uint16_t                                 m_encodeHeight;
    int                                      m_encodeType;
    unique_ptr<NvMedia2D, DestroyNvMedia2D>  m_upMedia2D;
    NvMedia2DComposeParameters               m_params;
    vector<NvSciBufObj>                      m_vInputSciBufObjs;
    unique_ptr<CCascadedMaster>              m_upCascadedMaster;
    unique_ptr<CBufferPool<CBuffer>>         m_upBufferPool;
    vector<NvSciBufObj>                      m_vOutputSciBufObjs;
    CBuffer*                                 m_pCurrentBuffer;
    HWNvmediaOutputPipelineContext*          m_poutputpipeline;
    NvSciSyncObj                             m_IEPSignalSyncObj = nullptr;

    //hw_video_sensorpipelinedatacbconfig_t* m_pcbconfig;
    //HWNvmediaEventHandlerRegDataCbConfig* m_peventhandlercbconfig;

    u32                                     _blockindex = 0;
    u32                                     _sensorindex = 0;
    INvSIPLClient::ConsumerDesc::OutputType _outputtype;

#if (HW_NVMEDIA_CHANGE_ME_LATER == 1)
    // may delete later
    uint8_t*                                m_pBuff = nullptr;
#endif
    hw_video_sensorpipelinedatacb           _datacb = nullptr;
    void*                                   _pcontext = nullptr;
    // record it down for bak, the user need data type
    HW_VIDEO_REGDATACB_TYPE                 _regdatacbtype;
    // set when do init test file operation
    HW_VIDEO_BUFFERFORMAT_MAINTYPE          _orinmaintype;
    // set when do init test file operation
    HW_VIDEO_BUFFERFORMAT_SUBTYPE           _orinsubtype;
    // set when do register data callback operation
    HW_VIDEO_BUFFERFORMAT_MAINTYPE          _usermaintype;
    // set when do register data callback operation
    HW_VIDEO_BUFFERFORMAT_SUBTYPE           _usersubtype;
    // 0 or 1
    u32                                     _bneedgetpixel = 0;
    // the expected origin sub type format
    HW_VIDEO_BUFFERFORMAT_SUBTYPE           _expectedoriginsubtype;
    // 0 or 1, 1 means has log that the _datacb is nullptr tip log
    u32                                     _blogdatacbnull = 0;
    // 0 or 1
    u32                                     _bsynccb = 1;


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
};
#endif
