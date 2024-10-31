/*
 * Copyright (c) Hozon SOC Co., Ltd. 2022-2022. All rights reserved.
 *
 * Description: transfer file normal task
 */
#ifndef UPDATE_NTTASK_TRANSFER_FILE_H_
#define UPDATE_NTTASK_TRANSFER_FILE_H_

#include <fstream>

#include "update_manager/taskbase/normal_task_base.h"
#include "update_manager/taskbase/task_object_def.h"
#include "update_manager/config/config_manager.h"

namespace hozon {
namespace netaos {
namespace update {

class UpdateNTTaskTransferFile : public NormalTaskBase {
public:
    UpdateNTTaskTransferFile(NormalTaskBase* pParent, STObject::TaskCB pfnCallback,
        const SensorInfo_t& sensorInfo, const UpdateCase_t& process, bool isTopTask = false);
    ~UpdateNTTaskTransferFile();

    uint8_t                 GetProgress();

    virtual uint32_t        doAction();
    virtual void            onCallbackAction(uint32_t result);

    uint32_t                StartToMemoryErase();
    void                    OnMemoryEraseResult(STTask *task, uint32_t result);

    uint32_t                StartToRequestBinDownload();
    void                    OnRequestBinDownloadResult(STTask *task, uint32_t result);

    uint32_t                StartToTransFile();
    void                    OnTransFileResult(STTask *task, uint32_t result);

    uint32_t                StartToRequestTransExit();
    void                    OnRequestTransExitResult(STTask *task, uint32_t result);

    uint32_t                StartToCheckDependency();
    void                    OnCheckDependencyResult(STTask *task, uint32_t result);

    uint32_t                StartToUpdateInstall();
    void                    OnUpdateInstallResult(STTask *task, uint32_t result);

    uint32_t                StartToCheckUpdateProgress();
    void                    OnCheckUpdateProgressResult(STTask *task, uint32_t result);

private:
    UpdateNTTaskTransferFile(const UpdateNTTaskTransferFile &);
    UpdateNTTaskTransferFile & operator = (const UpdateNTTaskTransferFile &);

    uint8_t                 CalcCrc8(const std::vector<uint8_t>& data, uint8_t crc);
    uint16_t                CalcCrc16(const std::vector<uint8_t>& data, uint16_t crc);

private:
    /// for transfer file
    SensorInfo_t            m_sensorInfo;
    UpdateCase_t            m_process;

    uint32_t                m_reqTotalSize;
    uint32_t                m_reqCompletedSize;
    uint16_t                m_reqBlockSize;
    uint8_t                 m_reqBlockIndex;
    uint16_t                m_reqCrc16;
    uint8_t*                m_buff;
    std::ifstream           ifs;
    uint8_t                 m_progress;
};

}  // namespace update
}  // namespace netaos
}  // namespace hozon
#endif  // UPDATE_NTTASK_TRANSFER_FILE_H_
