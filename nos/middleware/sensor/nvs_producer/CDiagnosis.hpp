#ifndef CAMERA_DIAGNOSIS_H
#define CAMERA_DIAGNOSIS_H

#include <stdio.h>
#include <vector>
#include <algorithm>
#include "CSensorRegInf.hpp"
#include "CustomInterface.hpp"
#include "diag_config.hpp"
#include "Cphm_client.h"

using namespace std;

class CameraDiagnosis {
public:
    CameraDiagnosis(CSensorRegInf* upsensorInf);
    ~CameraDiagnosis();
    void CameraDiagnosisInit(const uint32_t uSensor);
    void CameraDiagnosisStart();
    void CameraDiagnosisNotify(uint32_t sensorId, uint32_t cameraFaultId);
    void CameraDiagnosisSetInitFlag(uint8_t flag);

private:
    int Clear_reg(void);
    int Sensor_Safety_check(void);
    int Sensor_96717_check(void);
#if 0
    int PMIC_check(void);
#endif
    int Max96712_Safety_Setting(void);
    void Max96712_Safety_Readout();
    int Max20087_Diagnosis(void);
    void* diagnose_ThreadFunc();
    void CameraDiagnosisReportFault(uint32_t cameraFaultId, uint8_t cameraFaultObj);
    void CameraDiagnosisDesFault(uint32_t cameraFaultId, uint16_t error_status);
    void CameraDiagnosisPowerFault(uint8_t powerId, uint16_t error_status);

    std::vector<int> Camera_ids_group_a;
    std::vector<int> Camera_ids_group_b;
    std::vector<int> Camera_ids_group_c;
    std::vector<int> Camera_ids_group_d;
    std::vector<int> Camera_ids_group = std::vector<int>(4,-1);
    uint8_t get_interrupt_flag = 0;

    CSensorRegInf* m_upsensorInf;
    std::unique_ptr<std::thread> m_upThread = nullptr;
};

#endif  // #define CAMERA_DIAGNOSIS_H