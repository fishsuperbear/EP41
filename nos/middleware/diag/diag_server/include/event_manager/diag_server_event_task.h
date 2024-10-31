
#include "diag/common/include/thread_pool.h"

namespace hozon {
namespace netaos {
namespace diag {

class DiagServerClearDTCInfo : public BaseTask
{
public:
    DiagServerClearDTCInfo(const uint32_t dtcGroup)
    : m_dtcGroup(dtcGroup)
    {
    }

    virtual ~DiagServerClearDTCInfo()
    {
    }

    virtual int Run();
    uint32_t m_dtcGroup;
};

class DiagServerReportDTCNumByStatusMask : public BaseTask
{
public:
    DiagServerReportDTCNumByStatusMask(const uint8_t dtcStatusMask)
    : m_dtcStatusMask(dtcStatusMask)
    {
    }

    virtual ~DiagServerReportDTCNumByStatusMask()
    {
    }

    virtual int Run();
    uint8_t m_dtcStatusMask;
};

class DiagServerReportDTCByStatusMask : public BaseTask
{
public:
    DiagServerReportDTCByStatusMask(const uint8_t dtcStatusMask)
    : m_dtcStatusMask(dtcStatusMask)
    {
    }

    virtual ~DiagServerReportDTCByStatusMask()
    {
    }

    virtual int Run();
    uint8_t m_dtcStatusMask;
};

class DiagServerReportDTCSnapshotIdentification : public BaseTask
{
public:
    DiagServerReportDTCSnapshotIdentification()
    {
    }

    virtual ~DiagServerReportDTCSnapshotIdentification()
    {
    }

    virtual int Run();
};

class DiagServerReportDTCSnapshotRecordByDTCNumber : public BaseTask
{
public:
    DiagServerReportDTCSnapshotRecordByDTCNumber(const uint32_t dtcValue, const uint32_t number)
    : m_dtcValue(dtcValue)
    , m_number(number)
    {
    }

    virtual ~DiagServerReportDTCSnapshotRecordByDTCNumber()
    {
    }

    virtual int Run();
    uint32_t m_dtcValue;
    uint8_t m_number;
};

class DiagServerReportSupportedDTC : public BaseTask
{
public:
    DiagServerReportSupportedDTC()
    {
    }

    virtual ~DiagServerReportSupportedDTC()
    {
    }

    virtual int Run();
};

class DiagServerControlDTCStatusType : public BaseTask
{
public:
    DiagServerControlDTCStatusType(const DIAG_CONTROLDTCSTATUSTYPE& controlDtcStatusType)
    : m_controlDtcStatusType(controlDtcStatusType)
    {
    }

    virtual ~DiagServerControlDTCStatusType()
    {
    }

    virtual int Run();
    DIAG_CONTROLDTCSTATUSTYPE m_controlDtcStatusType;
};

class DiagServerReportDTCEvent : public BaseTask
{
public:
    DiagServerReportDTCEvent(uint32_t dtcValue, uint8_t dtcStatus)
    : m_dtcValue(dtcValue)
    , m_dtcStatus(dtcStatus)
    {
    }

    virtual ~DiagServerReportDTCEvent()
    {
    }

    virtual int Run();
    uint32_t m_dtcValue;
    uint8_t m_dtcStatus;
};

class DiagServerSessionChange : public BaseTask
{
public:
    DiagServerSessionChange(uint32_t sessionType)
    : m_sessionType(sessionType)
    {
    }

    virtual ~DiagServerSessionChange()
    {
    }

    virtual int Run();
    uint32_t m_sessionType;
};

class DiagServerRequestOutputDtcInfo : public BaseTask
{
public:
    DiagServerRequestOutputDtcInfo()
    {
    }

    virtual ~DiagServerRequestOutputDtcInfo()
    {
    }

    virtual int Run();
};

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
