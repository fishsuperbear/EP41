/**
 * @file DiagServerEventImpl.h
 * @brief Declaration file of class DiagServerEventImpl.
 */
#include "diag/diag_server/include/common/diag_server_def.h"
#include "diag/diag_server/include/event_manager/diag_server_event_pub.h"
#include "sqlite3.h"
#include <mutex>
#include <vector>
#include <memory>
#include <unordered_map>

namespace hozon {
namespace netaos {
namespace diag {

enum DIAG_CHECK_RESULT
{
    DIAG_CHECK_TRUE = 0,
    DIAG_CHECK_FALSE,
    DIAG_CHECK_ERROR,
};

enum DIAG_CLEAR_GROUP
{
    DIAG_CLEAR_GROUP_P = 0xFFF1FF,
    DIAG_CLEAR_GROUP_B = 0xFFF2FF,
    DIAG_CLEAR_GROUP_C = 0xFFF3FF,
    DIAG_CLEAR_GROUP_U = 0xFFF4FF,
    DIAG_CLEAR_GROUP_ALL = 0xFFFFFF,
};

enum DIAG_GROUP
{
    DIAG_GROUP_P = 0b00000000,
    DIAG_GROUP_C = 0b01000000,
    DIAG_GROUP_B = 0b10000000,
    DIAG_GROUP_U = 0b11000000,
};

template<typename T>
union DiagDidDataUnion
{
    struct DiagDidData
    {
        T value;
    } data;

    uint8_t dataArr[sizeof(T)];
};

class DiagServerEventStatus;
class DiagServerEventImpl
{
public:
    DiagServerEventImpl();

    virtual ~DiagServerEventImpl();

    DiagServerEventImpl(const DiagServerEventImpl& ref) = delete;

    DiagServerEventImpl& operator = (const DiagServerEventImpl& ref) = delete;

    static DiagServerEventImpl* getInstance();

    static void destroy();

    bool fileExists(const char* fileName);
    /*
    * new Circle,when diagservice restarted call this function
    */
    void newCircle();

    bool checkDbFileAndCreateTable();

    bool createFileAndTables();

    bool sqliteOperator(const char* file, const char *sql);

    bool clearDTCInformation(const uint32_t dtcGroup);

    bool reportDTCByStatusMask(const uint8_t dtcStatusMask, std::vector<DiagDtcData>& dtcInfos);

    bool reportDTCSnapshotIdentification(std::vector<DiagDtcData>& dtcInfos);

    bool reportDTCSnapshotRecordByDTCNumber(const uint32_t dtc, const uint16_t ssrNumber, std::vector<DiagDtcData>& dtcInfos);

    void reportSupportedDTC(std::vector<DiagDtcData>& dtcInfos);

    bool reportDTCEvent(uint32_t faultKey, uint8_t faultStatus);

    bool queryDtcStatus(sqlite3* db, uint32_t dtc, DiagServerEventStatus& cDtcStatus);

    bool dealWithDtcRecover(DiagServerEventStatus& cDtcStatus, char* szTemp);

    bool dealWithDtcOccur(DiagServerEventStatus& cDtcStatus, char* szTemp);

    bool getInsertDtcSql(uint32_t dtcValue, uint8_t iDtcstatus, uint8_t iTripcount, uint8_t iCurFaultObj, uint64_t finalFaultObj, char* szTemp);

    bool queryDtcDb(std::vector<DiagDtcData>& dtcInfos, char* sqlBuf, const int nByte);

    bool querySnapshotRecordByDTCNumber(std::vector<DiagDtcData>& dtcInfos, char* sqlBuf, const int nByte);

    bool getAgingDtcs(std::unordered_map<uint32_t, uint32_t>& outAgingDtcs);

    void requestOutputDtcInfo();

    void deleteAllDtc();

    void deleteGroupDtc(std::string& group);

    void getGroupDtc(std::string& group, std::vector<DiagDtcData>& dtcInfos);

    void getAllDtc(std::vector<DiagDtcData>& dtcInfos);

    bool checkDbEmpty();

    void notifyDtcControlSetting(const DIAG_CONTROLDTCSTATUSTYPE& controlDtcStatusType);

    void setPubHandle(std::shared_ptr<DiagServerEventPub> spDiagServerEventPub);

    void fillSsrData(DiagDtcData& dtcdata);

    void getExtendData(sqlite3* db, uint32_t faultKey);

    uint8_t getVoltageData();

private:
    static std::mutex m_instanceMtx;
    static std::mutex s_syncWriteMtx;
    static DiagServerEventImpl* s_instance;
    static bool s_bCreate;

    bool m_dbEmptyCheckedFlag = false;
    bool m_dbEmpty = false;
    std::shared_ptr<DiagServerEventPub> m_spEventPub;
};

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
