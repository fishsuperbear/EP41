/*
 * Copyright (c) hozonauto. 2021-2022. All rights reserved.
 * Description: TspPkiConfig class definition.
 */

#ifndef V2C_TSP_PKI_TSP_PKI_CERT_MANAGE_H
#define V2C_TSP_PKI_TSP_PKI_CERT_MANAGE_H

#include <string>
#include <mutex>
#include <string>
#include <thread>
#include <future>
#include "curl/curl.h"
#include <condition_variable>
#include "tsp_pki_def.h"
#include "tsp_pki_log.h"
#include "https_types.h"
#include "phm/include/phm_client.h" 
#include "cfg/include/config_param.h"
#include "devm/include/devm_device_info.h"
#include "crypto_adapter.h"


// #include "tsp_pki_persistency.h"

namespace hozon {
namespace netaos {
namespace tsp_pki {
    
enum FmFaultCode : uint16_t {
    FmPresetCertsError = 4720,      // yaml config file load error
    FmVinError = 4730,         // Vin error
    FmSnError = 4740,          // sn error
    // FmCertExpireError = 8755,  // cert out of life
    // FmCertInvalidError = 8756,  // cert invalid
    FmCertUnableRecoverError = 4750,  // device cert invalid and can not recover
    FmCfgFileError = 4760  // device cert invalid and can not recover
};

class TspPkiCertManage {
public:
    enum CertStatus {
        kCertStatusOk = 0,       // normal status
        kCertStatusInThreshold,  // in threshold,can use,need to update
        kCertStatusNone,         // no cert
        kCertStatusUpdating,     // during updating or applying
        kCertStatusUnkown,       // unable to get the time status of cert
        kCertStatusExpire,       // Over time
        kCertStatusBeforActive,  // current time before the active time of cert
        kCertStatusCnUnmatch,    // the vin and sn unmatch the cn of cert
    };

    struct TspPkiErrorInfo
    {
        int key;
        std::string info;
    };
    const std::string cfg_key_pki_vin_ = "pki/vin";
    const std::string cfg_key_pki_sn_ = "pki/sn";
    const std::string cfg_key_pki_privatekey_slot_ = "pki/slot";
    const std::string cfg_key_pki_ca_path_ = "pki/root_ca_path";
    const std::string cfg_key_pki_device_cert_path_ = "pki/device_cert_path";
    const std::string cfg_key_pki_device_cert_status_ ="pki/status";

    static TspPkiCertManage& Instance();
    static void Destroy();

    void Start();
    void Stop();

    TspPkiError GetPkiApplyStatus();
    std::string GetClientCertPath();
    std::string GetKeySlot();
    std::string GetNextSlot();
    // JITPARA& GetJitPara(){
    //     return jit_para_;
    // };
    std::string GetVin(){
        return vin_;
    };
    std::string GetSn(){
        return sn_;
    };
    enum CertStatus GetCertStatus();
    std::string GetErrInfo(TspPkiError errNum);


private:
    enum CertValidity {
        CertValidity_Valid = 0,
        CertValidity_CnNotMatch,
        CertValidity_OutofTimeValidity,
        CertValidity_TimeNotSync,
        CertValidity_KeyNotMatch,
        CertValidity_CertVerifyFail,
        CertValidity_CertInvalid,
        CertValidity_FileNotExist,
        CertValidity_Unknown
    };

    enum ApplyType {
        First_Time = 0,
        Second_Time_or_More,
    };

    bool stopped_ = false;
    uint8_t apply_pki_status_ = TspPki_OtherError;
    CertStatus cert_status_;
    std::mutex cert_mutex_;
    std::mutex tsync_mutex_;
    std::mutex vin_mutex_;
    std::mutex cryptoConfig_mutex_;
    std::thread work_th_;
    std::thread work_tsync_;
    std::string cur_slot_;
    // JITPARA jit_para_;
    bool time_synchronized_ = false;
    std::shared_ptr<std::promise<std::string>> vinPromise_ = std::make_shared<std::promise<std::string>>();
    std::future<std::string> vinFuture_;

    std::string vin_ = "";
    std::string sn_ = "";
    std::string defaultVin = "FFFFFFFFFFFFFFFFF";
    std::condition_variable conTsync_;
    static TspPkiCertManage* instance_;
    bool tsyncbyManual_ = false;
    // TspPkiPersistency presetCert_;
    std::map<uint8_t,std::string> errInfoMap_;
    // std::shared_ptr<phm::PHMClient> spPHMClient_;

    TspPkiCertManage();
    TspPkiCertManage(const TspPkiCertManage& certM){};
    ~TspPkiCertManage();

    bool CheckClientCert();
    static void TimeSyncThread();
    TspPkiError ExtractCertFromResponse(const std::string& body, std::string& cert_pem);
    bool InstallCryptoConfig();
    int ApplyClientCert(ApplyType applyType);
    int ApplyCert(ApplyType applyType);
    std::string GetCsr(const std::string& vin,const std::string& sn);
    static void vinCallback(const std::string& clientName, const std::string& paramName, const std::string& paramValue, const uint8_t& paramType);
    static void tsynCallback(const std::string& clientName, const std::string& paramName, const std::string& paramValue, const uint8_t& paramType);
    int GetVinFromCfg();
    int GetSnFromCfg();
    int SetVin(const std::string& inVIN);
    std::string GetCertSubjectName();
    std::string GetCertCN();
    int compareCN(const std::string& vin,const std::string& sn);
    int DeletePreCert();
    int BakPresetCert();
    bool IsPfxexist();
    int ExpireOrBeforActiveHandle();
    int CnUnmatchHandle();
    int ImportCert_PEM(std::string cert_pem);
    template<typename T>
    int PublishCfgValue(std::string key,T state);
    int PublishAllinfo();
    std::string ParseSnFromJson(std::string json_file);
    std::string ParseVinFromJson(std::string json_file);
    hozon::netaos::cfg::ConfigParam *cfg_param_ = nullptr;
    hozon::netaos::devm::DevmClientDeviceInfo device_info_;
    hozon::netaos::crypto::CryptoAdapter cryAdp_;

};

}
}
}

#endif