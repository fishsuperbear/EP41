#include <iostream>
#include "routine_control.h"
#include "lidar_calibration.h"

RoutineControl::RoutineControl()
: DiagAgentRoutineControl()
{
}

RoutineControl::~RoutineControl()
{
}

bool
RoutineControl::Start(const uint16_t rid, const std::vector<uint8_t>& reqData, std::vector<uint8_t>& resData)
{
    std::cout << "RoutineControl::Start rid: " << rid << std::endl;
    bool bResult = false;
    switch (rid) {
    case 0xD500:
        {
            resData.push_back(0xD5);
            resData.push_back(0x00);
            resData.emplace_back(0x02);  // success: 0x02    failed: 0x05
            bResult = true;
        }
        break;
    case 0xFC88:
        {
            resData.push_back(0xFC);
            resData.push_back(0x88);
            std::vector<uint8_t> data;
            bResult = LidarCalibration::getInstance()->NotifyCalibrationMode(reqData, data);
            if (data.size() > 0) {
                resData.emplace_back(data[0]);  // success: 0x02    failed: 0x05
            }
        }
        break;
    case 0xFD08:
        {
            resData.push_back(0xFD);
            resData.push_back(0x08);
            std::vector<uint8_t> data;
            bResult = LidarCalibration::getInstance()->CheckPrecondition(reqData, data);
            if (data.size() > 0) {
                resData.emplace_back(data[0]);  // success: 0x02    failed: 0x05
            }
        }
        break;
    case 0xFD10:
        {
            resData.push_back(0xFD);
            resData.push_back(0x10);
            std::vector<uint8_t> data;
            bResult = LidarCalibration::getInstance()->StartCalibrateLidar(reqData, data);
            if (data.size() > 0) {
                resData.emplace_back(data[0]);  // success: 0x02    failed: 0x05
            }
        }
        break;
    default:
        break;
    }

    return bResult;
}

bool
RoutineControl::Stop(const uint16_t rid, const std::vector<uint8_t>& reqData, std::vector<uint8_t>& resData)
{
    std::cout << "RoutineControl::Stop rid: " << rid << std::endl;
    bool bResult = false;
    switch (rid)
    {
        case 0xD500:
            resData.push_back(0xD5);
            resData.push_back(0x00);
            resData.emplace_back(0x02);  // success: 0x02    failed: 0x05
            bResult = true;
            break;
        default:
            break;
    }

    return bResult;
}

bool
RoutineControl::Result(const uint16_t rid, const std::vector<uint8_t>& reqData, std::vector<uint8_t>& resData)
{
    std::cout << "RoutineControl::Result rid: " << rid << std::endl;
    bool bResult = false;
    switch (rid) {
    case 0xD500:
        {
            resData.push_back(0xD5);
            resData.push_back(0x00);
            resData.emplace_back(0x01);
            bResult = true;
        }
        break;
    case 0xFC88:
        {
            resData.push_back(0xFC);
            resData.push_back(0x88);
            std::vector<uint8_t> data;
            bResult = LidarCalibration::getInstance()->NotifyCalibrationModeResult(reqData, data);
            if (data.size() > 0) {
                for (uint i = 0; i < data.size(); ++i) {
                    resData.emplace_back(data[i]);
                }
            }
        }
        break;
    case 0xFD08:
        {
            resData.push_back(0xFD);
            resData.push_back(0x08);
            std::vector<uint8_t> data;
            bResult = LidarCalibration::getInstance()->CheckPreconditionResult(reqData, data);
            if (data.size() > 0) {
                for (uint i = 0; i < data.size(); ++i) {
                    resData.emplace_back(data[i]);
                }
            }
        }
        break;
    case 0xFD10:
        {
            resData.push_back(0xFD);
            resData.push_back(0x10);
            std::vector<uint8_t> data;
            bResult = LidarCalibration::getInstance()->StartCalibrateLidarResult(reqData, data);
            if (data.size() > 0) {
                for (uint i = 0; i < data.size(); ++i) {
                    resData.emplace_back(data[i]);
                }
            }
        }
        break;
    default:
        break;
    }

    return bResult;
}
