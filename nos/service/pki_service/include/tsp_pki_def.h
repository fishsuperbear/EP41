/*
 * Copyright (c) hozonauto. 2021-2022. All rights reserved.
 * Description: TspPkiConfig class definition.
 */

#ifndef V2C_TSP_PKI_TSP_PKI_DEF_H
#define V2C_TSP_PKI_TSP_PKI_DEF_H

#include <string>
#include <mutex>

namespace hozon {
namespace netaos {
namespace tsp_pki {

enum TspPkiError:uint8_t {

    TspPki_Success = 0x00, // 成功

    // 表示失败的code范围:
    // 0x10 ~ 0x1f, 表示MDC端错误:
    TspPki_ConfigError = 0x10, // 配置参数错误
    TspPki_SnError = 0x11, // SN错误
    TspPki_VinError = 0x12, // VIN错误
    TspPki_PresetCertError = 0x13, // 预置证书/私钥错误
    TspPki_SystemTimeError = 0x14, // 系统时间错误
    TspPki_RootCertError = 0x15, // 根证书错误
    TspPki_ImportFailed = 0x16, // 证书导入失败
    TspPki_GenerateP10Failed = 0x17, // p10生成失败
    TspPki_DeviceCertAbnormal = 0x18, // 正式设备证书异常
    TspPki_CnUnequal = 0x19, // VIN或SN和已有证书的VIN和SN不匹配
    TspPki_OtherError = 0x1f, // 其他错误

    // 0x20 ~ 0x2f, 表示通信错误:
    TspPki_ComError = 0x20, // 通信错误
    TspPki_ComTimeout = 0x21, // 通信超时
    TspPki_TlsError = 0x22, // tls错误

    // 0x30 ~ 0x3f ，表示服务器错误
    TspPki_HttpError = 0x30, // http错误
    TspPki_ServerError = 0x31, // 服务器错误

    // 0x40 ~ 0x4f ，表示PKI申请状态
    TspPki_WaitVin = 0x40, // 等待VIN码
    TspPki_WaitTimeSync = 0x41, // 等待时间同步

    TspPki_Canceled = 0xf1,  // For internal usage.
};

enum TspComError {
    TspCom_Success = 0x00,
    TspCom_HttpErrorStart = 100,       // http error start.

    TspCom_ClientCertInvalid = 0xff01,
    TspCom_SystemParamInvalid,
    TspCom_Timeout,
    TspCom_ComError,
    TspCom_OtherError,
    TspCom_Cancelled
};

}
}
}

#endif