#ifndef DIAG_SERVER_CONFIG_H
#define DIAG_SERVER_CONFIG_H

#include <mutex>
#include <unordered_map>
#include <string>

#include "diag/common/include/data_def.h"
#include "diag/diag_server/include/common/diag_server_def.h"

namespace hozon {
namespace netaos {
namespace diag {

const uint8_t DIAG_SERVER_ID = 0x01;

class DiagServerConfig {
public:
    static DiagServerConfig* getInstance();

    void Init();

    void DeInit();

    void LoadDiagConfig();
    DiagConfigInfo GetDiagConfigInfo() {return diag_config_info_;}

    // Software
    uint16_t GetDiagServerPhysicAddress(const uint8_t serverId = DIAG_SERVER_ID);
    bool IsUpdateManagerAddress(const uint16_t address, const uint8_t serverId = DIAG_SERVER_ID);
    bool IsRemoteAddress(const uint16_t address, const uint8_t serverId = DIAG_SERVER_ID);
    bool IsUpdateManagerOrRemoteAddress(const uint16_t address, const uint8_t serverId = DIAG_SERVER_ID);
    bool IsSupportDoip(const uint8_t serverId = DIAG_SERVER_ID);
    bool IsSupportDoCan(const uint8_t serverId = DIAG_SERVER_ID);
    bool IsSupportDoSomeip(const uint8_t serverId = DIAG_SERVER_ID);

    // External Service
    bool QueryAllExternalService(std::vector<std::string>& service);
    bool QueryExternalServiceBySid(const uint8_t sid, std::vector<std::string>& service);
    bool QueryExternalServiceByDid(const uint16_t did, std::vector<std::string>& service, bool bWrite = false);
    bool QueryExternalServiceByRid(const uint16_t rid, std::vector<std::string>& service);

    // Data Transfer
    uint64_t GetDiagServerDataTransferSize(const uint8_t configId);
    bool QueryDataTransferConfig(const uint8_t configId, DiagTransferConfigDataInfo& configInfo);

    // Security  Level
    uint32_t GetDiagServerSecurityMask(const uint8_t levelId);

    // Permission
    bool QueryAccessPermissionBySid(const DiagServerServiceRequestOpc sid, DiagAccessPermissionDataInfo& accessInfo);
    bool QueryAccessPermissionBySidAndSubFunc(const DiagServerServiceRequestOpc sid, const uint8_t subFuc, DiagAccessPermissionDataInfo& accessInfo);
    bool QueryReadAccessPermissionByDid(const uint16_t did, DiagAccessPermissionDataInfo & accessInfo);
    bool QueryWriteAccessPermissionByDid(const uint16_t did, DiagAccessPermissionDataInfo & accessInfo);
    bool QueryAccessPermissionByRid(const uint16_t rid, DiagAccessPermissionDataInfo& accessInfo);
    bool QuerySecurityLevelInfoByName(const uint8_t levelId, DiagSecurityLevelDataInfo& levelInfo);
    bool QueryAccessPermissionByName(const uint8_t accessId, DiagAccessPermissionDataInfo& accessInfo);

    // SID
    uint16_t GetSidMaxPendingNum(const DiagServerServiceRequestOpc sid);
    bool QuerySidSupport(const DiagServerServiceRequestOpc sid, const DiagTargetAddressType& addrType);
    bool QuerySidSupportInActiveSession(const DiagServerServiceRequestOpc sid, const uint16_t sourceAddr);
    bool QuerySubFunctionSupportInActiveSession(const DiagServerServiceRequestOpc sid, const uint8_t subFunc);
    bool QuerySubFunctionSupportForSid(const DiagServerServiceRequestOpc sid, const uint8_t subFunc, const DiagTargetAddressType& addrType);
    bool QuerySidHaveSubFunction(const DiagServerServiceRequestOpc sid);
    bool QuerySubFunctionSupportSuppressPosMsgindication(const DiagServerServiceRequestOpc sid, const uint8_t subFunc);

    // DID
    uint16_t QueryDidDataSize(const uint16_t did);

    // RID
    bool QueryRidSupport(const uint16_t rid);
    bool QueryRidMultiStartSupport(const uint16_t rid);
    bool QueryRidSupportList(std::vector<uint16_t>& ridList);
    bool QueryRidDataInfo(const uint16_t rid, DiagRidDataInfo& dataInfo);
    bool QueryRidDataLenBySubFunction(const uint16_t rid, const uint8_t subFunc, const size_t length, const bool isReply);

    // Session
    uint16_t QuerySessionP2Timer(const DiagServerSessionCode& session);
    uint16_t QuerySessionP2StarTimer(const DiagServerSessionCode& session);
    uint16_t QuerySessionS3Timer(const DiagServerSessionCode& session);

    // Dem
    void GetAllDtc(std::vector<uint32_t>& allDtc);
    void GetDemInfo(DiagDemDataInfo& demInfo);
    void GetSingleDtcData(const uint32_t faultKey, DiagDtcDataInfo& outDtcData);
    void GetSingleDtcData(const uint32_t dtc, const uint8_t curFaultObj, DiagDtcDataInfo& outDtcData);
    void GetAllFaultOfDtc(const uint32_t dtc, std::vector<uint32_t>& outAllFault);
    bool CheckFaultIsExist(uint32_t faultKey);
    bool CheckDtcIsExist(const uint32_t dtc);

    // For Test
    void QueryPrintConfigData();

private:
    bool LoadDMAndDTCConfig();

private:
    DiagServerConfig();
    DiagServerConfig(const DiagServerConfig &);
    DiagServerConfig & operator = (const DiagServerConfig &);
private:
    static DiagServerConfig* instance_;
    static std::mutex mtx_;

    // diag config info
    DiagConfigInfo diag_config_info_;

    // unordered_map<softwareClusterId, DiagSoftWareClusterDataInfo>
    std::unordered_map<uint8_t, DiagSoftWareClusterDataInfo> diag_softwarecluster_info_map_;

    // unordered_map<externalServiceConfigId, DiagExternalServiceConfigDataInfo>
    std::unordered_map<uint8_t, DiagExternalServiceConfigDataInfo> diag_external_service_config_info_map_;

    // unordered_map<transferconfigId, DiagTransferConfigDataInfo>
    std::unordered_map<uint8_t, DiagTransferConfigDataInfo> diag_transfer_config_info_map_;

    // unordered_map<sessionId, DiagSessionDataInfo>
    std::unordered_map<uint8_t, DiagSessionDataInfo> diag_session_info_map_;

    // unordered_map<securityLevelId, DiagSecurityLevelDataInfo>
    std::unordered_map<uint8_t, DiagSecurityLevelDataInfo> diag_securitylevel_info_map_;

    // unordered_map<accessPermissionId, DiagAccessPermissionDataInfo>
    std::unordered_map<uint8_t, DiagAccessPermissionDataInfo> diag_accesspermission_info_map_;

    // unordered_map<sidId, DiagSidDataInfo>
    std::unordered_map<uint8_t, DiagSidDataInfo> diag_sid_info_map_;

    // unordered_map<didId, DiagDidDataInfo>
    std::unordered_map<uint16_t, DiagDidDataInfo> diag_did_info_map_;

    // unordered_map<ridId, DiagRidDataInfo>
    std::unordered_map<uint16_t, DiagRidDataInfo> diag_rid_info_map_;

    // unordered_map<faultKey, DiagDtcDataInfo>
    std::unordered_map<uint32_t, DiagDtcDataInfo> diag_dtc_info_map_;
    std::unordered_multimap<uint32_t, uint32_t> diag_dtc_faultkey_multimap_;

    DiagDemDataInfo m_diagDemDataInfo;
};

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
#endif  // DIAG_SERVER_CONFIG_H
