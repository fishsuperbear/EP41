#include "update_manager/common/um_functions.h"
#include "update_manager/taskbase/task_object_def.h"

namespace hozon {
namespace netaos {
namespace update {

std::string GetDesayUpdateResultString(const int32_t& result)
{
    switch (result)
    {
        case 0x00:
            return "OTA_SUCCESS";
        case 0xE0:
            return "BSL_UNKNOWN_ERROR_TYPE";
        case 0xE1:
            return "OTA_RECV_TIMEOUT";
        case 0xE2:
            return "OTA_ADDR_NOT_VALID";
        case 0xE3:
            return "OTA_OPEN_FAILED";
        case 0xE4:
            return "OTA_MD5_CHECK_FAILED";
        case 0xE5:
            return "OTA_SOCK_FAILED";
        case 0xE6:
            return "OTA_BIND_FAILED";
        case 0xF4:
            return "BSL_FORMAT_ERROR";
        case 0xF5:
            return "BSL_STATUS_IDEL";
        case 0xF6:
            return "BSL_STATUS_BUSY";
        case 0xF7:
            return "BSL_NOT_ENOUGH_MEMORY";
        case 0xF8:
            return "BSL_SIZE_ERROR";
        case 0xF9:
            return "BSL_FLASH_ERROR";
        case 0xFA:
            return "BSL_COUNT_ERROR";
        case 0xFB:
            return "BSL_SN_ERROR";
        case 0xFC:
            return "BSL_ERASE_ERROR";
        case 0xFD:
            return "BSL_ADDRESS_ERROR";
        case 0xFE:
            return "BSL_MODE_ERROR";
        case 0xFF:
            return "BSL_BLOCK_TYPE_ERROR";
        default:
            return "NULL";
    }
}

std::string GetDesayUpdateStatusString(const int32_t& status)
{
    switch (status)
    {
        case 0x00:
            return "UPDATESTATE_IDLE";
        case 0x01:
            return "UPDATESTATE_BEGIN";
        case 0x02:
            return "UPDATESTATE_IN_PROGRESS";
        case 0x03:
            return "UPDATESTATE_FAILED";
        case 0x04:
            return "UPDATESTATE_SUCCESS";
        default:
            return "NULL";
    }
}

std::string GetDesayUpdateString(const int32_t& result)
{
    switch (result)
    {
        case 0x00:
            return "E_OK";
        case 0x01:
            return "E_NOT_OK";
        default:
            return "NULL";
    }
}

std::string GetDesayUpdateCurPartitonString(const int32_t& slot)
{
    switch (slot)
    {
        case 0x01:
            return "OTA_CURRENT_SLOT_A";
        case 0x02:
            return "OTA_CURRENT_SLOT_B";
        case 0x03:
            return "E_NOT_OK";
        default:
            return "NULL";
    }
}

std::string GetTaskResultString(const uint32_t& taskResult)
{
    N_Result_t res = static_cast<N_Result_t>(taskResult);
    switch (res)
    {
        case N_Result_t::N_OK:
            return "N_OK";
        case N_Result_t::N_FIRST:
            return "N_FIRST";
        case N_Result_t::N_ERROR:
            return "N_ERROR";
        case N_Result_t::N_TIMEOUT_P2_CLIENT:
            return "N_TIMEOUT_P2_CLIENT";
        case N_Result_t::N_TIMEOUT_P2START_CLIENT:
            return "N_TIMEOUT_P2START_CLIENT";
        case N_Result_t::N_TIMEOUT_P3_CLIENT_PYH:
            return "N_TIMEOUT_P3_CLIENT_PYH";
        case N_Result_t::N_TIMEOUT_P3_CLIENT_FUNC:
            return "N_TIMEOUT_P3_CLIENT_FUNC";
        case N_Result_t::N_WRONG_SN:
            return "N_WRONG_SN";
        case N_Result_t::N_UNEXP_PDU:
            return "N_UNEXP_PDU";
        case N_Result_t::N_WFT_OVRN:
            return "N_WFT_OVRN";
        case N_Result_t::N_BUFFER_OVFLW:
            return "N_BUFFER_OVFLW";
        case N_Result_t::N_RX_ON:
            return "N_RX_ON";
        case N_Result_t::N_WRONG_PARAMETER:
            return "N_WRONG_PARAMETER";
        case N_Result_t::N_WRONG_VALUE:
            return "N_WRONG_VALUE";
        case N_Result_t::N_USER_CANCEL:
            return "N_USER_CANCEL";
        case N_Result_t::N_WAIT:
            return "N_WAIT";
        case N_Result_t::N_RETRY_TIMES_LIMITED:
            return "N_RETRY_TIMES_LIMITED";
        case N_Result_t::N_NRC:
            return "N_NRC";
        default:
            return "NULL";
    }
}

}  // namespace update
}  // namespace netaos
}  // namespace hozon
