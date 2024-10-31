#ifndef HW_NVMEDIA_DEF_IMPL_H
#define HW_NVMEDIA_DEF_IMPL_H

#include "hw_nvmedia_baseinc_impl.h"

/*
 * You should fix the logic later.
 */
#define HW_NVMEDIA_FIX_ME_LATER 1
/*
 * You should change the logic later.
 */
#define HW_NVMEDIA_CHANGE_ME_LATER 1
/*
 * You may change the logic later.
 */
#define HW_NVMEDIA_MAY_CHANGE_ME_LATER 1
/*
 * You may change the marked logic later.
 */
#define HW_NVMEDIA_MAY_CHANGE_ME_LATER_MARKED 0
/*
 * You may change the logic later.
 */
#define HW_NVMEDIA_MAY_DELETE_ME_LATER 1


enum INTERNAL_CHECK_HW_NVMEDIA_USE_CONSUMER {
    USE_CUDA_CONSUMER,
    USE_ENCODER_CONSUMER,
    USE_COMMON_CONSUMER,
    USE_VIC_CONSUMER,
    /*
     * First consumer: Cuda
     * Second consumer: Encoder(may not use)
     * Third consumer: Common consumer
     * The total number count is HW_NVMEDIA_NUM_CONSUMERS.
     */
    HW_NVMEDIA_NUM_CONSUMERS,
};

/*
 * 1 means basic output buffer in file like name testfile_common0_0.yuv.
 * For common consumer use only.
 */
#define HW_NVMEDIA_COMMON_CONSUMER_BASIC_TESTRAWOUTPUT 0

/*
 * 1 means do not use NvSciBufObjGetPixels function so that to enhance performance.
 * 0 means use NvSciBufObjGetPixels function.
 * Curently can not discard (the output yuv file cannnot change to correct jpeg picture).
 * The isp0 output is bl format.
 */
#define HW_NVMEDIA_COMMON_CONSUMER_FURTHER_TESTRAWOUTPUT 0

/*
 * 1 means do not use NvSciBufObjGetPixels function so that to enhance performance.
 * 0 means use NvSciBufObjGetPixels function.
 */
#define HW_NVMEDIA_COMMON_CONSUMER_DISCARD_GETPIXELS 0

/*
 * 1 means basic output buffer in file like name testfile_common0_0.yuv.
 * For vic consumer use only.
 */
#define HW_NVMEDIA_COMMON_CONSUMER_BASIC_TESTRAWOUTPUT_VIC 0

/*
 * Currently, we only define the nvmedia implement environment s32 return defines.
 * The total value is 0 mean totally SUCCESS.
 * Once the return value is not 0, we do not change it though function call return.
 * When other value, according to the following bit list:
 * bit[31:30](severity)	0:Success, 1:Informational, 2:Warning, 3:Error
 * bit[29:29](custom)	0:Use hresult/ntstatus defines, 1:Use custom define. Now is 1.
 * bit[28:28](ntstatus)	0:Not use ntstatus define. 1:Use ntstatus define(valid when bit29 is 0)
 * bit[27:16](facility)	12bits to tag the facility
 * bit[15:0](code)		16bits code correspondent to the specific facility
 */
typedef s32 hw_ret_s32;

const char* hw_ret_s32_getdesc(hw_ret_s32 i_s32);

/*
 * The CHECK_LOG_HW_RET_S32 macro need log implementation, so define it in log impl head
 * file.
 */
#define CHK_HW_RET_S32(ret) \
    do {                    \
        if (ret != 0) {     \
            return ret;     \
        }                   \
    } while (0)

enum HW_RET_S32_FACILITY {
    /*
     * We reserved the 0 value for future use.
     */
    HW_RET_S32_FACILITY_MIN = 1,
    HW_RET_S32_FACILITY_MINMINUSONE = HW_RET_S32_FACILITY_MIN - 1,

    /*
     * The hw hal of nvmedia video type.
     * See details of HW_RET_S32_CODE_HW_HAL_NVMEDIA.
     */
    HW_RET_S32_FACILITY_HW_HAL_NVMEDIA,
    /*
     * See SIPLStatus defines in NvSIPLCommon.hpp.
     */
    HW_RET_S32_FACILITY_NVMEDIA_SIPLSTATUS,
    /*
     * See NvSciError defines in nvscierror.h.
     */
    HW_RET_S32_FACILITY_NVMEDIA_NVSCISTATUS,

    HW_RET_S32_FACILITY_MAXADDONE,
    HW_RET_S32_FACILITY_MAX = HW_RET_S32_FACILITY_MAXADDONE - 1,
};

enum HW_RET_S32_CODE_HW_HAL_NVMEDIA {
    /*
     * We reserved the 0 for no error. When 0 the : sentence use the different condition correspondent sentence.
     */
    HW_RET_S32_CODE_HW_HAL_NVMEDIA_MIN = 1,
    HW_RET_S32_CODE_HW_HAL_NVMEDIA_MINMINUSONE = HW_RET_S32_CODE_HW_HAL_NVMEDIA_MIN - 1,

    HW_RET_S32_CODE_HW_HAL_NVMEDIA_COMMON_UNEXPECTED,
    HW_RET_S32_CODE_HW_HAL_NVMEDIA_CHECK_VERSION_FAIL,
    HW_RET_S32_CODE_HW_HAL_NVMEDIA_PIPELINE_CONFIG_WRONG,
    HW_RET_S32_CODE_HW_HAL_NVMEDIA_BLOCKPIPELINE_NOT_ALWAYS_ENABLE_INNERHANDLE,
    HW_RET_S32_CODE_HW_HAL_NVMEDIA_SENSORPIPELINE_NOT_ALWAYS_ENABLE_INNERHANDLE,
    HW_RET_S32_CODE_HW_HAL_NVMEDIA_DEVICE_ALREADY_IN_USE,
    HW_RET_S32_CODE_HW_HAL_NVMEDIA_DEVICE_CLOSE_EXCHANGE_FAIL,
    HW_RET_S32_CODE_HW_HAL_NVMEDIA_PIPELINE_ALREADY_IN_USE,
    HW_RET_S32_CODE_HW_HAL_NVMEDIA_PIPELINE_CLOSE_EXCHANGE_FAIL,
    HW_RET_S32_CODE_HW_HAL_NVMEDIA_CONNECT_DID_NOT_RECEIVE_CONNECTED_EVENT,
    HW_RET_S32_CODE_HW_HAL_NVMEDIA_RECONCILE_EVENTHANDLER_NOT_RUNNING,
    HW_RET_S32_CODE_HW_HAL_NVMEDIA_BCANCALLPIPELINESTART_UNEXPECTED_NOT_0,
    HW_RET_S32_CODE_HW_HAL_NVMEDIA_BCANCALLPIPELINESTART_UNEXPECTED_NOT_1,
    HW_RET_S32_CODE_HW_HAL_NVMEDIA_REGDATACBTYPE_UNEXPECTED,
    HW_RET_S32_CODE_HW_HAL_NVMEDIA_REGDATACBTYPE_REGISTER_TWICE,
    HW_RET_S32_CODE_HW_HAL_NVMEDIA_NO_CORRESPONDENT_OUTPUTTYPE,
    HW_RET_S32_CODE_HW_HAL_NVMEDIA_OUTPUTTYPE_REGISTER_TWICE,
    HW_RET_S32_CODE_HW_HAL_NVMEDIA_OUTPUTTYPE_NOT_ENABLE_BUT_REGISTER_DATACB,
    HW_RET_S32_CODE_HW_HAL_NVMEDIA_ONLY_SUPPORT_DIRECTCB_MODE,
    HW_RET_S32_CODE_HW_HAL_NVMEDIA_ONLY_SUPPORT_SYNCCB_MODE,
    HW_RET_S32_CODE_HW_HAL_NVMEDIA_COMMON_CONSUMER_SUBTYPE_NOT_EXPECTED,
    HW_RET_S32_CODE_HW_HAL_NVMEDIA_COMMON_CONSUMER_UNKNOWN_SUBTYPE,

    HW_RET_S32_CODE_HW_HAL_NVMEDIA_MAXADDONE,
    HW_RET_S32_CODE_HW_HAL_NVMEDIA_MAX = HW_RET_S32_CODE_HW_HAL_NVMEDIA_MAXADDONE - 1,
};

// currently we has only two severity: 0 or Error, we may add Informational or Warning in the future.
#define HW_RET_S32_TAG_SEVERITY_ERROR (3 << 30)
#define HW_RET_S32_TAG_CUSTOM (1 << 29)
#define HW_RET_S32_GETTAG_CUSTOM(v) ((v)&HW_RET_S32_TAG_CUSTOM)
#define HW_RET_S32_IS_CUSTOM(v) (HW_RET_S32_GETTAG_CUSTOM(v) > 0)
#define HW_RET_S32_TAG_HW_HAL_NVMEDIA (HW_RET_S32_TAG_SEVERITY_ERROR | HW_RET_S32_TAG_CUSTOM | (HW_RET_S32_FACILITY_HW_HAL_NVMEDIA << 16))
#define HW_RET_S32_TAG_NVMEDIA_SIPLSTATUS (HW_RET_S32_TAG_SEVERITY_ERROR | HW_RET_S32_TAG_CUSTOM | (HW_RET_S32_FACILITY_NVMEDIA_SIPLSTATUS << 16))
#define HW_RET_S32_TAG_NVMEDIA_NVSCISTATUS (HW_RET_S32_TAG_SEVERITY_ERROR | HW_RET_S32_TAG_CUSTOM | (HW_RET_S32_FACILITY_NVMEDIA_NVSCISTATUS << 16))
#define HW_RET_S32_GET_FACILITY(v) (((v)&0xFFF0000) >> 16)
#define HW_RET_S32_GET_CODE(v) ((v)&0xFFFF)

#define HW_RET_S32_HW_HAL_NVMEDIA(code) ((code == 0) ? 0 : (HW_RET_S32_TAG_HW_HAL_NVMEDIA | code))
#define HW_RET_S32_NVMEDIA_SIPLSTATUS(siplstatus) ((siplstatus == NVSIPL_STATUS_OK) ? 0 : (HW_RET_S32_TAG_NVMEDIA_SIPLSTATUS | siplstatus))
#define HW_RET_S32_NVMEDIA_NVSCISTATUS(nvscistatus) ((nvscistatus == NvSciError_Success) ? 0 : (HW_RET_S32_TAG_NVMEDIA_NVSCISTATUS | nvscistatus))

/*
 * You can use C++ style here. Implement use only. Will not be included by user.
 */

enum HW_NVMEDIA_APPTYPE {
    HW_NVMEDIA_APPTYPE_MIN = 0,
    HW_NVMEDIA_APPTYPE_MINMINUSONE = HW_NVMEDIA_APPTYPE_MIN - 1,

    HW_NVMEDIA_APPTYPE_SINGLE_PROCESS,
    HW_NVMEDIA_APPTYPE_IPC_PRODUCER,
    HW_NVMEDIA_APPTYPE_IPC_CONSUMER_CUDA,
    HW_NVMEDIA_APPTYPE_IPC_CONSUMER_ENC,

    HW_NVMEDIA_APPTYPE_MAXADDONE,
    HW_NVMEDIA_APPTYPE_MAX = HW_NVMEDIA_APPTYPE_MAXADDONE - 1,
};

/*
 * The mode is definite when open the device.
 * Currently, the mode info is definite and can not be changed!
 * Para means parameters.
 */
class HWNvmediaDeviceOpenPara {
public:
    /*
     * It is recommended to use mail box mode(another mode is fifo).
     * busemailbox is 0 or 1. 1 mean use mail box mode.
     */
    void Init(HW_NVMEDIA_APPTYPE i_apptype, const char* i_platformname, const char* i_maskstr,
              u32 i_busemailbox, const char* i_nitofolderpath);

public:
    HW_NVMEDIA_APPTYPE apptype = HW_NVMEDIA_APPTYPE_SINGLE_PROCESS;
    const char* platformname = "";  // string like "V1SIM728S1RU3120NB20_CPHY_x4"
    const char* maskstr = "";       // mask string like "0x0001 0 0 0"
    std::vector<u32> vmasks;
    u32 busemailbox;
    const char* nitofolderpath = "";
};

#endif
