#include <iostream>
#include "lidar_calibration.h"

LidarCalibration* LidarCalibration::instance_ = nullptr;
std::mutex LidarCalibration::mtx_;
static bool g_check_flag = false;
static bool g_calibrate_flag = false;

LidarCalibration::LidarCalibration()
: is_calibrate_(ModeSwitch_Not_Start)
, check_result_(CheckResult_Failed)
, calibrate_result_(CalibrateResult_Init)
{
}

LidarCalibration*
LidarCalibration::getInstance()
{
    if (nullptr == instance_) {
        std::lock_guard<std::mutex> lck(mtx_);
        if (nullptr == instance_) {
            instance_ = new LidarCalibration();
        }
    }

    return instance_;
}

LidarCalibration::~LidarCalibration()
{
}

bool
LidarCalibration::NotifyCalibrationMode(const std::vector<uint8_t>& reqData, std::vector<uint8_t>& resData)
{
    std::cout << "LidarCalibration::NotifyCalibrationMode " << std::endl;
    is_calibrate_ = ModeSwitch_Progressing;
    uint32_t mode = reqData[0] << 16 | reqData[1] << 8 | reqData[2];
    if (mode == 0x00040301) {
        if (NotifyStateMachine(true)) {
            is_calibrate_ = ModeSwitch_Success;
            resData.push_back(0x02);
        }
        else {
            is_calibrate_ = ModeSwitch_Failed;
            resData.push_back(0x05);
        }

        return true;
    }

    return false;
}

bool
LidarCalibration::NotifyCalibrationModeResult(const std::vector<uint8_t>& reqData, std::vector<uint8_t>& resData)
{
    std::cout << "LidarCalibration::NotifyCalibrationModeResult " << std::endl;
    resData.clear();
    if (is_calibrate_ == ModeSwitch_Progressing) {
        resData.push_back(0x01);
    }
    else if (is_calibrate_ == ModeSwitch_Success) {
        resData.push_back(0x02);
    }
    else if (is_calibrate_ == ModeSwitch_Failed) {
        resData.push_back(0x03);
    }
    else{
        resData.push_back(0x00);
    }

    return true;
}

bool
LidarCalibration::CheckPrecondition(const std::vector<uint8_t>& reqData, std::vector<uint8_t>& resData)
{
    std::cout << "LidarCalibration::CheckPrecondition " << std::endl;
    resData.clear();
    if (is_calibrate_ == ModeSwitch_Success) {
        if (reqData[0] == 0x04 && reqData[1] == 0x04) {
            g_check_flag = true;
            check_result_ = CheckResult_Ok;
            resData.push_back(0x02);
        }
        else {
            check_result_ = CheckResult_Abnormal;
            resData.push_back(0x05);
        }

        return true;
    }
    check_result_ = CheckResult_Failed;
    return false;
}

bool
LidarCalibration::CheckPreconditionResult(const std::vector<uint8_t>& reqData, std::vector<uint8_t>& resData)
{
    std::cout << "LidarCalibration::CheckPreconditionResult " << std::endl;
    resData.clear();
    if (!g_check_flag) {
        check_result_ = CheckResult_Failed;
    }

    if (check_result_ == CheckResult_Abnormal) {
        resData.push_back(0x01);
        resData.push_back(0x00);
        resData.push_back(0x00);

        return true;
    }
    else if (check_result_ == CheckResult_Ok) {
        resData.push_back(0x02);
        resData.push_back(0x00);
        resData.push_back(0x00);

        return true;
    }
    else if (check_result_ == CheckResult_Failed) {
        resData.push_back(0x04);
        resData.push_back(0x00);
        resData.push_back(0x00);

        return true;
    }
    else {
        resData.push_back(0x03);
        resData.push_back(0x00);
        resData.push_back(0x00);
        return true;
    }

    return false;
}

bool
LidarCalibration::StartCalibrateLidar(const std::vector<uint8_t>& reqData, std::vector<uint8_t>& resData)
{
    std::cout << "LidarCalibration::StartCalibrateLidar " << std::endl;
    resData.clear();
    if (is_calibrate_ == ModeSwitch_Success) {
        if (reqData[0] == 0x04 && reqData[1] == 0x04) {
            g_calibrate_flag = true;
            calibrate_result_ = CalibrateResult_Success;
            resData.push_back(0x02);
        }
        else {
            calibrate_result_ = CalibrateResult_Failed;
            resData.push_back(0x05);
        }

        return true;
    }

    return false;
}

bool
LidarCalibration::StartCalibrateLidarResult(const std::vector<uint8_t>& reqData, std::vector<uint8_t>& resData)
{
    std::cout << "LidarCalibration::StartCalibrateLidarResult " << std::endl;
    resData.clear();
    if (!g_calibrate_flag) {
        calibrate_result_ = CalibrateResult_Failed;
    }

    if (calibrate_result_ == CalibrateResult_Init) {
        resData.push_back(0x00);
        resData.push_back(0x00);
        return true;
    }
    else if (calibrate_result_ == CalibrateResult_Success) {
        resData.push_back(0x02);
        resData.push_back(0x00);
        return true;
    }
    else if (calibrate_result_ == CalibrateResult_Failed) {
        resData.push_back(0x03);
        resData.push_back(0x00);
        return true;
    }
    else {
        resData.push_back(0x01);
        resData.push_back(0x00);
        return true;
    }

    return false;
}

bool
LidarCalibration::GetCalibrateResult(std::vector<uint8_t>& resData)
{
    std::cout << "LidarCalibration::GetCalibrateResult " << std::endl;
    resData.clear();
    if (calibrate_result_ == CalibrateResult_Failed) {
        for (int i = 0; i < 40; ++i) {
            resData.push_back(0x00);
        }

        return true;
    }
    return false;
}

bool
LidarCalibration::NotifyStateMachine(bool status)
{
    return status;
}
