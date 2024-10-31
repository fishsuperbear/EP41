#ifndef LIDAR_CALIBRATION_H
#define LIDAR_CALIBRATION_H

#include <mutex>
#include <vector>
// #include "diag/diag_agent/include/service/diag_agent_routine_control.h"

// using namespace hozon::netaos::diag::diag_agent;

class LidarCalibration {

public:
    enum ModeSwitchStatus {
        ModeSwitch_Not_Start        = 0x00,
        ModeSwitch_Progressing      = 0x01,
        ModeSwitch_Success          = 0x02,
        ModeSwitch_Failed           = 0x03,
    };

    enum CheckResult {
        CheckResult_Abnormal        = 0x01,
        CheckResult_Ok              = 0x02,
        CheckResult_Executing       = 0x03,
        CheckResult_Failed          = 0x04,
    };

    enum CalibrateResult {
        CalibrateResult_Init             = 0x00,
        CalibrateResult_Progressing      = 0x01,
        CalibrateResult_Success          = 0x02,
        CalibrateResult_Failed           = 0x03,
    };

    static LidarCalibration* getInstance();
    virtual ~LidarCalibration();

    bool NotifyCalibrationMode(const std::vector<uint8_t>& reqData, std::vector<uint8_t>& resData);
    bool NotifyCalibrationModeResult(const std::vector<uint8_t>& reqData, std::vector<uint8_t>& resData);
    bool CheckPrecondition(const std::vector<uint8_t>& reqData, std::vector<uint8_t>& resData);
    bool CheckPreconditionResult(const std::vector<uint8_t>& reqData, std::vector<uint8_t>& resData);
    bool StartCalibrateLidar(const std::vector<uint8_t>& reqData, std::vector<uint8_t>& resData);
    bool StartCalibrateLidarResult(const std::vector<uint8_t>& reqData, std::vector<uint8_t>& resData);
    bool GetCalibrateResult(std::vector<uint8_t>& resData);

private:
    bool NotifyStateMachine(bool status);

private:
    LidarCalibration();
    LidarCalibration(const LidarCalibration &);
    LidarCalibration & operator = (const LidarCalibration &);

    static LidarCalibration* instance_;
    static std::mutex mtx_;
    ModeSwitchStatus is_calibrate_;
    CheckResult check_result_;
    CalibrateResult calibrate_result_;
};

#endif  // LIDAR_CALIBRATION_H
