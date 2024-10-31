/*
 * Copyright (c) Hozon SOC Co., Ltd. 2022-2022. All rights reserved.
 *
 * Description: transfer file normal task
 */

#include "update_nttask_transfer_file.h"
#include "update_cttask_command.h"
#include "update_manager/log/update_manager_logger.h"
#include "update_manager/record/ota_record.h"

namespace hozon {
namespace netaos {
namespace update {


UpdateNTTaskTransferFile::UpdateNTTaskTransferFile(NormalTaskBase* pParent, STObject::TaskCB pfnCallback,
    const SensorInfo_t& sensorInfo, const UpdateCase_t& process, bool isTopTask)
    : NormalTaskBase(OTA_NTTASK_TRANSFER_FILE, pParent, pfnCallback, isTopTask)
    , m_sensorInfo(sensorInfo)
    , m_process(process)
    , m_reqTotalSize(0)
    , m_reqCompletedSize(0)
    , m_reqBlockSize(0)
    , m_reqBlockIndex(0)
    , m_reqCrc16(0xFFFF)
    , m_buff(nullptr)
    , m_progress(process.beginProgress)
{
}

UpdateNTTaskTransferFile::~UpdateNTTaskTransferFile()
{
    if (ifs.is_open()) {
        ifs.close();
    }
    if (nullptr != m_buff) {
        delete []m_buff;
        m_buff = nullptr;
    }
}

uint8_t UpdateNTTaskTransferFile::GetProgress()
{
    return m_progress;
}

uint32_t UpdateNTTaskTransferFile::doAction()
{
    if (m_process.fileType == 1) {
        // boot firmware may not need memory erase just tranfer data
        return StartToRequestBinDownload();
    }
    else if (m_process.fileType == 2 || m_process.fileType == 3) {
        // app or cal firmware need memory erase first
        return StartToMemoryErase();
    }
    else {
        UPDATE_LOG_E("Invalid file type!~");
        return N_ERROR;
    }

}

void UpdateNTTaskTransferFile::onCallbackAction(uint32_t result)
{
    if (N_OK == result) {
        m_progress = m_process.endProgress;
    }
}

uint32_t UpdateNTTaskTransferFile::StartToMemoryErase()
{
    UPDATE_LOG_D("StartToMemoryErase $31 01 FF 00, ecu: %s, progress: %d%%!~", m_sensorInfo.name.c_str(), m_progress);
    OTARecoder::Instance().RecordStepStart(m_sensorInfo.name, "Memory Erase", m_progress);
    //  $31 01 FF 00
    TaskReqInfo reqInfo;
    TaskResInfo resInfo;
    reqInfo.reqUpdateType = m_sensorInfo.updateType;
    reqInfo.reqSa = UpdateSettings::Instance().UmLogicAddr();
    reqInfo.reqTa = m_sensorInfo.logicalAddr;
    reqInfo.reqWaitTime = 0;

    if (reqInfo.reqUpdateType == 1) {
        reqInfo.reqContent = { 0x31, 0x01, 0xFF, 0x00, (uint8_t)(m_process.memoryAddr >> 24), (uint8_t)(m_process.memoryAddr >> 16),
            (uint8_t)(m_process.memoryAddr >> 8), (uint8_t)(m_process.memoryAddr), (uint8_t)(m_process.memorySize >> 24),
            (uint8_t)(m_process.memorySize >> 16), (uint8_t)(m_process.memorySize >> 8), (uint8_t)(m_process.memorySize) };
    }
    else if (reqInfo.reqUpdateType == 2) {
        reqInfo.reqContent = { 0x31, 0x01, 0xFF, 0x00, 0x44, (uint8_t)(m_process.memoryAddr >> 24), (uint8_t)(m_process.memoryAddr >> 16),
            (uint8_t)(m_process.memoryAddr >> 8), (uint8_t)(m_process.memoryAddr), (uint8_t)(m_process.memorySize >> 24),
            (uint8_t)(m_process.memorySize >> 16), (uint8_t)(m_process.memorySize >> 8), (uint8_t)(m_process.memorySize) };
    }

    reqInfo.reqExpectContent = { 0x71, 0x01, 0xFF, 0x00, 0x02 };

    UpdateCTTaskCommand* task = new UpdateCTTaskCommand(this,
                                            CAST_TASK_CB(&UpdateNTTaskTransferFile::OnMemoryEraseResult),
                                            reqInfo, resInfo);
    return post(task);
}

void UpdateNTTaskTransferFile::OnMemoryEraseResult(STTask *task, uint32_t result)
{
    m_progress = m_progress + 1 <= 100 ? m_progress + 1 : 100;
    UPDATE_LOG_D("OnMemoryEraseResult ecu: %s, result: %d, progress: %d%%!~", m_sensorInfo.name.c_str(), result, m_progress);
    OTARecoder::Instance().RecordStepFinish(m_sensorInfo.name, "Memory Erase", result, m_progress);
    if (N_OK == result) {
        result = StartToRequestBinDownload();
        if (eContinue != result) {
            onCallbackResult(result);
        }
    }
    else {
        if (eContinue != result) {
            onCallbackResult(result);
        }
    }
}

uint32_t UpdateNTTaskTransferFile::StartToRequestBinDownload()
{
    UPDATE_LOG_D("StartToRequestBinDownload $34 ecu: %s, progress: %d%%!~", m_sensorInfo.name.c_str(), m_progress);
    OTARecoder::Instance().RecordStepStart(m_sensorInfo.name, "Request Bin Download", m_progress);
    // $34 00 44 memoryAddr memorySize
    TaskReqInfo reqInfo;
    TaskResInfo resInfo;
    reqInfo.reqUpdateType = m_sensorInfo.updateType;
    reqInfo.reqSa = UpdateSettings::Instance().UmLogicAddr();
    reqInfo.reqTa = m_sensorInfo.logicalAddr;
    reqInfo.reqWaitTime = 0;

    reqInfo.reqContent = { 0x34, 0x00, 0x44, (uint8_t)(m_process.memoryAddr >> 24), (uint8_t)(m_process.memoryAddr >> 16),
        (uint8_t)(m_process.memoryAddr >> 8), (uint8_t)(m_process.memoryAddr), (uint8_t)(m_process.memorySize >> 24),
        (uint8_t)(m_process.memorySize >> 16), (uint8_t)(m_process.memorySize >> 8), (uint8_t)(m_process.memorySize) };
    reqInfo.reqExpectContent = { 0x74 };

    UpdateCTTaskCommand* task = new UpdateCTTaskCommand(this,
                                            CAST_TASK_CB(&UpdateNTTaskTransferFile::OnRequestBinDownloadResult),
                                            reqInfo, resInfo);
    return post(task);

}

void UpdateNTTaskTransferFile::OnRequestBinDownloadResult(STTask *task, uint32_t result)
{
    m_progress = m_progress + 1 <= 100 ? m_progress + 1 : 100;
    UPDATE_LOG_D("OnRequestBinDownloadResult ecu: %s, result: %d, progress: %d%%!~", m_sensorInfo.name.c_str(), result, m_progress);
    OTARecoder::Instance().RecordStepFinish(m_sensorInfo.name, "Request Bin Download", result, m_progress);
    if (N_OK == result) {
        UpdateCTTaskCommand* sftask = static_cast<UpdateCTTaskCommand*>(task);
        if (nullptr == sftask) {
            result = N_ERROR;
            onCallbackResult(result);
            return;
        }

        if (sftask->GetResInfo().resContent.size() == 4 || sftask->GetResInfo().resContent.size() == 6) {
            m_reqBlockSize = (sftask->GetResInfo().resContent[sftask->GetResInfo().resContent.size() - 2] << 8)
                | (sftask->GetResInfo().resContent[sftask->GetResInfo().resContent.size() - 1]);
        }
        else {
            UPDATE_LOG_E("Invalid $74 data length!~");
            result = N_ERROR;
            onCallbackResult(result);
            return;
        }

        // ifstream read update file, get file size
        ifs.open(m_process.filePath, std::ifstream::in | std::ifstream::binary);
        ifs.seekg(0, std::ios::end);
        m_reqTotalSize = ifs.tellg();
        m_reqCompletedSize = 0;
        ifs.seekg(0, std::ios::beg);
        m_buff = new uint8_t[m_reqTotalSize + 1];
        memset(m_buff, 0x00, m_reqTotalSize + 1);

        OTARecoder::Instance().RecordStepStart(m_sensorInfo.name, "Transfer Bin File", m_progress);
        result = StartToTransFile();
        if (eContinue != result) {
            onCallbackResult(result);
        }
    }
    else {
        if (eContinue != result) {
            onCallbackResult(result);
        }
    }
}

uint32_t UpdateNTTaskTransferFile::StartToTransFile()
{
    UPDATE_LOG_D("StartToTransFile $36, ecu: %s, BlockIndex: %d, BlockSize: %d, CompletedSize: %d / TotalSize: %d(%d%%), progress: %d%%!~",
        m_sensorInfo.name.c_str(), m_reqBlockIndex, m_reqBlockSize, m_reqCompletedSize, m_reqTotalSize, (m_reqCompletedSize * 100)/m_reqTotalSize, m_progress);
    OTARecoder::Instance().RecordStepStart(m_sensorInfo.name, "", m_progress);
    TaskReqInfo  reqInfo;
    TaskResInfo  resInfo;
    ifs.read(reinterpret_cast<char*>(m_buff + m_reqCompletedSize), m_reqBlockSize - 2);
    uint32_t readBytes = ifs.gcount();
    UPDATE_LOG_D("ecu: %s, readBytes: %d.", m_sensorInfo.name.c_str(), readBytes);
    ++m_reqBlockIndex;

    reqInfo.reqUpdateType = m_sensorInfo.updateType;
    reqInfo.reqSa = UpdateSettings::Instance().UmLogicAddr();
    reqInfo.reqTa = m_sensorInfo.logicalAddr;
    reqInfo.reqWaitTime = 10*1000;
    reqInfo.reqContent = { 0x36, m_reqBlockIndex };
    reqInfo.reqContent.insert(reqInfo.reqContent.end(), m_buff + m_reqCompletedSize, m_buff + (m_reqCompletedSize + readBytes));
    reqInfo.reqExpectContent = { 0x76, m_reqBlockIndex };
    UpdateCTTaskCommand* task = new UpdateCTTaskCommand(this,
                                            CAST_TASK_CB(&UpdateNTTaskTransferFile::OnTransFileResult),
                                            reqInfo, resInfo);


    m_reqCompletedSize += readBytes;
    return post(task);
}

void UpdateNTTaskTransferFile::OnTransFileResult(STTask *task, uint32_t result)
{
    m_progress = (m_process.beginProgress + 2) + (m_process.endProgress - m_process.beginProgress - 4) * m_reqCompletedSize / m_reqTotalSize;
    UPDATE_LOG_D("OnTransFileResult ecu: %s, result: %d, progress: %d%%!~", m_sensorInfo.name.c_str(), result, m_progress);
    if (N_OK == result) {
        UpdateCTTaskCommand* sftask = static_cast<UpdateCTTaskCommand*>(task);
        if (nullptr == sftask) {
            result = N_ERROR;
            onCallbackResult(result);
            return;
        }

        if (m_reqCompletedSize >= m_reqTotalSize) {
            UPDATE_LOG_D("File transfer completed success, ecu: %s, progress: %d%%!~", m_sensorInfo.name.c_str(), m_progress);
            OTARecoder::Instance().RecordStepFinish(m_sensorInfo.name, "Transfer Bin File", result, m_progress);
            // all data send completed.
            ifs.close();
            result = StartToRequestTransExit();
        }
        else {
            // continue transfer file
            result = StartToTransFile();
        }

        if (eContinue != result) {
            onCallbackResult(result);
        }
    }
    else {
        OTARecoder::Instance().RecordStepFinish(m_sensorInfo.name, "Transfer Bin File", result, m_progress);
        if (eContinue != result) {
            onCallbackResult(result);
        }
    }
}

uint32_t UpdateNTTaskTransferFile::StartToRequestTransExit()
{
    UPDATE_LOG_D("StartToRequestTransExit $37, ecu: %s, progress: %d%%!~", m_sensorInfo.name.c_str(), m_progress);
    OTARecoder::Instance().RecordStepStart(m_sensorInfo.name, "Request Transfer Exit", m_progress);
    // $37  request transfer exit.
    TaskReqInfo reqInfo;
    TaskResInfo resInfo;
    reqInfo.reqUpdateType = m_sensorInfo.updateType;
    reqInfo.reqSa = UpdateSettings::Instance().UmLogicAddr();
    reqInfo.reqTa = m_sensorInfo.logicalAddr;
    reqInfo.reqWaitTime = 0;

    reqInfo.reqContent = { 0x37 };
    reqInfo.reqExpectContent = { 0x77 };

    UpdateCTTaskCommand* task = new UpdateCTTaskCommand(this,
                                            CAST_TASK_CB(&UpdateNTTaskTransferFile::OnRequestTransExitResult),
                                            reqInfo, resInfo);
    return post(task);
}

void UpdateNTTaskTransferFile::OnRequestTransExitResult(STTask *task, uint32_t result)
{
    m_progress = m_progress + 1 <= 100 ? m_progress + 1 : 100;
    UPDATE_LOG_D("OnRequestTransExitResult ecu: %s, result: %d, progress: %d%%!~", m_sensorInfo.name.c_str(), result, m_progress);
    OTARecoder::Instance().RecordStepFinish(m_sensorInfo.name, "Request Transfer Exit", result, m_progress);
    if (N_OK == result) {
        UpdateCTTaskCommand* sftask = static_cast<UpdateCTTaskCommand*>(task);
        if (nullptr == sftask) {
            result = N_ERROR;
            onCallbackResult(result);
            return;
        }

        if (m_process.fileType == 2 || m_process.fileType == 3) {
            // app or cal firmware need check dependency, boot firmwate may not need
            result = StartToCheckDependency();
        }
        if (eContinue != result) {
            onCallbackResult(result);
        }
    }
    else {
        if (eContinue != result) {
            onCallbackResult(result);
        }
    }
}

uint32_t UpdateNTTaskTransferFile::StartToCheckDependency()
{
    m_progress = m_progress + 1 <= 100 ? m_progress + 1 : 100;
    UPDATE_LOG_D("StartToCheckDependency $31 01 FF 01, ecu: %s, progress: %d%%!~", m_sensorInfo.name.c_str(), m_progress);
    OTARecoder::Instance().RecordStepStart(m_sensorInfo.name, "Check Dependency", m_progress);
    // $31 01 FF 01
    TaskReqInfo reqInfo;
    TaskResInfo resInfo;
    reqInfo.reqUpdateType = m_sensorInfo.updateType;
    reqInfo.reqSa = UpdateSettings::Instance().UmLogicAddr();
    reqInfo.reqTa = m_sensorInfo.logicalAddr;
    reqInfo.reqWaitTime = 15000 * 6;   // CRC16 need wait time at least 15s such as Lidar

    m_reqCrc16 = CalcCrc16(std::vector<uint8_t>(m_buff, m_buff + m_reqTotalSize), m_reqCrc16);

    reqInfo.reqContent = { 0x31, 0x01, 0xFF, 0x01, (uint8_t)(m_process.memoryAddr >> 24), (uint8_t)(m_process.memoryAddr >> 16),
        (uint8_t)(m_process.memoryAddr >> 8), (uint8_t)(m_process.memoryAddr), (uint8_t)(m_process.memorySize >> 24),
        (uint8_t)(m_process.memorySize >> 16), (uint8_t)(m_process.memorySize >> 8), (uint8_t)(m_process.memorySize),
        (uint8_t)(m_reqCrc16 >> 8), (uint8_t)m_reqCrc16 };
    reqInfo.reqExpectContent = { 0x71, 0x01, 0xFF, 0x01, 0x02 };

    UpdateCTTaskCommand* task = new UpdateCTTaskCommand(this,
                                            CAST_TASK_CB(&UpdateNTTaskTransferFile::OnCheckDependencyResult),
                                            reqInfo, resInfo);
    return post(task);
}

void UpdateNTTaskTransferFile::OnCheckDependencyResult(STTask *task, uint32_t result)
{
    UPDATE_LOG_D("OnCheckDependencyResult ecu: %s, result: %d, m_progress: %d%%!~", m_sensorInfo.name.c_str(), result, m_progress);
    OTARecoder::Instance().RecordStepFinish(m_sensorInfo.name, "Check Dependency", result, m_progress);
    if (!m_sensorInfo.havaFileSystem) {
        onCallbackResult(result);
        return;
    }

    result = StartToUpdateInstall();
    if (eContinue != result) {
        onCallbackResult(result);
    }

}

uint32_t UpdateNTTaskTransferFile::StartToUpdateInstall()
{
    return N_OK;
}

void UpdateNTTaskTransferFile::OnUpdateInstallResult(STTask *task, uint32_t result)
{

}

uint32_t UpdateNTTaskTransferFile::StartToCheckUpdateProgress()
{
    return N_OK;
}

void UpdateNTTaskTransferFile::OnCheckUpdateProgressResult(STTask *task, uint32_t result)
{
}



uint8_t UpdateNTTaskTransferFile::CalcCrc8(const std::vector<uint8_t>& data, uint8_t crc)
{
    uint8_t crc8 = crc;
    for (auto it: data) {
        crc8 += it;
    }
    return crc8;
}

/*********************************************************
 *
 * The checksum algorithm to be used shall be the CRC16-CITT:
 *  - Polynomial: x^16+x^12+x^5+1 (1021 hex)
 *  - Initial value: FFFF (hex)
 *  For a fast CRC16-CITT calculation a look-up table implementation is the preferred solution. For ECUs with a
 *  limited amount of flash memory (or RAM), other implementations may be necessary.
 *  Example 1: crc16-citt c-code (fast)
 *  This example uses a look-up table with pre-calculated CRCs for fast calculation.
 * ******************************************************/
uint16_t UpdateNTTaskTransferFile::CalcCrc16(const std::vector<uint8_t>& data, uint16_t crc)
{
    /*Here is crctab[256], this array is fixed */
    uint16_t crctab[256] =
    {
        0x0000, 0x1021, 0x2042, 0x3063, 0x4084, 0x50A5, 0x60C6, 0x70E7,
        0x8108, 0x9129, 0xA14A, 0xB16B, 0xC18C, 0xD1AD, 0xE1CE, 0xF1EF,
        0x1231, 0x0210, 0x3273, 0x2252, 0x52B5, 0x4294, 0x72F7, 0x62D6,
        0x9339, 0x8318, 0xB37B, 0xA35A, 0xD3BD, 0xC39C, 0xF3FF, 0xE3DE,
        0x2462, 0x3443, 0x0420, 0x1401, 0x64E6, 0x74C7, 0x44A4, 0x5485,
        0xA56A, 0xB54B, 0x8528, 0x9509, 0xE5EE, 0xF5CF, 0xC5AC, 0xD58D,
        0x3653, 0x2672, 0x1611, 0x0630, 0x76D7, 0x66F6, 0x5695, 0x46B4,
        0xB75B, 0xA77A, 0x9719, 0x8738, 0xF7DF, 0xE7FE, 0xD79D, 0xC7BC,
        0x48C4, 0x58E5, 0x6886, 0x78A7, 0x0840, 0x1861, 0x2802, 0x3823,
        0xC9CC, 0xD9ED, 0xE98E, 0xF9AF, 0x8948, 0x9969, 0xA90A, 0xB92B,
        0x5AF5, 0x4AD4, 0x7AB7, 0x6A96, 0x1A71, 0x0A50, 0x3A33, 0x2A12,
        0xDBFD, 0xCBDC, 0xFBBF, 0xEB9E, 0x9B79, 0x8B58, 0xBB3B, 0xAB1A,
        0x6CA6, 0x7C87, 0x4CE4, 0x5CC5, 0x2C22, 0x3C03, 0x0C60, 0x1C41,
        0xEDAE, 0xFD8F, 0xCDEC, 0xDDCD, 0xAD2A, 0xBD0B, 0x8D68, 0x9D49,
        0x7E97, 0x6EB6, 0x5ED5, 0x4EF4, 0x3E13, 0x2E32, 0x1E51, 0x0E70,
        0xFF9F, 0xEFBE, 0xDFDD, 0xCFFC, 0xBF1B, 0xAF3A, 0x9F59, 0x8F78,
        0x9188, 0x81A9, 0xB1CA, 0xA1EB, 0xD10C, 0xC12D, 0xF14E, 0xE16F,
        0x1080, 0x00A1, 0x30C2, 0x20E3, 0x5004, 0x4025, 0x7046, 0x6067,
        0x83B9, 0x9398, 0xA3FB, 0xB3DA, 0xC33D, 0xD31C, 0xE37F, 0xF35E,
        0x02B1, 0x1290, 0x22F3, 0x32D2, 0x4235, 0x5214, 0x6277, 0x7256,
        0xB5EA, 0xA5CB, 0x95A8, 0x8589, 0xF56E, 0xE54F, 0xD52C, 0xC50D,
        0x34E2, 0x24C3, 0x14A0, 0x0481, 0x7466, 0x6447, 0x5424, 0x4405,
        0xA7DB, 0xB7FA, 0x8799, 0x97B8, 0xE75F, 0xF77E, 0xC71D, 0xD73C,
        0x26D3, 0x36F2, 0x0691, 0x16B0, 0x6657, 0x7676, 0x4615, 0x5634,
        0xD94C, 0xC96D, 0xF90E, 0xE92F, 0x99C8, 0x89E9, 0xB98A, 0xA9AB,
        0x5844, 0x4865, 0x7806, 0x6827, 0x18C0, 0x08E1, 0x3882, 0x28A3,
        0xCB7D, 0xDB5C, 0xEB3F, 0xFB1E, 0x8BF9, 0x9BD8, 0xABBB, 0xBB9A,
        0x4A75, 0x5A54, 0x6A37, 0x7A16, 0x0AF1, 0x1AD0, 0x2AB3, 0x3A92,
        0xFD2E, 0xED0F, 0xDD6C, 0xCD4D, 0xBDAA, 0xAD8B, 0x9DE8, 0x8DC9,
        0x7C26, 0x6C07, 0x5C64, 0x4C45, 0x3CA2, 0x2C83, 0x1CE0, 0x0CC1,
        0xEF1F, 0xFF3E, 0xCF5D, 0xDF7C, 0xAF9B, 0xBFBA, 0x8FD9, 0x9FF8,
        0x6E17, 0x7E36, 0x4E55, 0x5E74, 0x2E93, 0x3EB2, 0x0ED1, 0x1EF0
    };

    uint16_t crc16 = crc;
    uint16_t tmp = 0;
    for (auto it : data) {
        tmp = (crc16 >> 8) ^ it;
        crc16 = (crc16 << 8) ^ crctab[tmp];
    }

    return crc16;
}


}  // namespace update
}  // namespace netaos
}  // namespace hozon