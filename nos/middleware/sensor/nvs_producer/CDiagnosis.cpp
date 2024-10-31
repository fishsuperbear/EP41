/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
#include "CDiagnosis.hpp"

unsigned int camera_self_buf[] = {0x4f09, 0x4f0a, 0x4f0b, 0x460c, 0x460d, 0x460e, 0x460f};
uint8_t camera_data[2][20] = {0};

unsigned int camera_96717_buf[] = {0x1d, 0x13, 0x1f, 0x112, 0x1d13, 0x1b, 0x1d20, 0x1f, 0x1d13};
uint8_t camera_96717_data[2][20] = {0};

#if 0
unsigned int PMIC_buf[]={0x7E,0xDA,0x7A,0x78,0x80};
uint8_t PMIC_data[2][20]={0};
#endif

#define MAX_VAILD_SENSORS_NUM 12
// 0-前视120°，1-前视30°，2-后视
// 4-左前，5-左后，6-右后，7-右前 //周视
// 8-前，9-左，10-右，11-后 //环视
uint8_t faultIndexArray[MAX_VAILD_SENSORS_NUM] = {1U, 2U, 3U, 0U, 8U, 10U, 11U, 9U, 4U, 5U, 6U, 7U};

CameraDiagnosis::CameraDiagnosis(CSensorRegInf* upsensorInf) {
    m_upsensorInf = upsensorInf;
    CameraPhmClient::getInstance()->Init();
}

CameraDiagnosis::~CameraDiagnosis(void) {
    CameraPhmClient::getInstance()->DeInit();
    if (m_upThread != nullptr) {
        m_upThread->join();
        m_upThread.reset();
    }
}

int CameraDiagnosis::Clear_reg(void) {
    uint8_t temp = 0;
    for (auto id : Camera_ids_group)  //four group  = four max96712
    {
        if (id < 0 || id > 15) {
            continue;
        }
        for (size_t i = 0; i < 7; i++) {
            m_upsensorInf->InterfaceReadRegister(id, CAMERA_REG_DESER, Max96712_Clear_Reg[i], &temp, 1);
        }

        if (get_interrupt_flag) {
            for (size_t i = 0; i < 8; i++) {
                m_upsensorInf->InterfaceReadRegister(id, CAMERA_REG_DESER, Max96712_Interrupt_Clear_Reg[i], &temp, 1);
            }
        }
    }
    return 0;
}

int CameraDiagnosis::Sensor_Safety_check(void) {
    uint8_t read_status = 0;
    memset(camera_data, 0, sizeof(camera_data));
    for (auto id : Camera_ids_group_a) {
        if (id < 0 || id > 15) {
            continue;
        }
        if (id == 0) {
            for (size_t i = 0; i < 7; i++) {
                read_status =
                    m_upsensorInf->InterfaceReadRegister(id, CAMERA_REG_SENSOR, camera_self_buf[i], &camera_data[0][i], 1);
                if (read_status != 0) {
                    camera_data[0][i] = 0;
                }
            }
        } else if (id == 1) {
            for (size_t j = 0; j < 7; j++) {
                read_status =
                    m_upsensorInf->InterfaceReadRegister(id, CAMERA_REG_SENSOR, camera_self_buf[j], &camera_data[1][j], 1);
                if (read_status != 0) {
                    camera_data[1][j] = 0;
                }
            }
        }
    }
    for (auto id : Camera_ids_group_a) {
        if ((id == 0) || (id == 1)) {
            Sensor_diagnose(&camera_data);
        }
    }
    return 0;
}

int CameraDiagnosis::Sensor_96717_check(void) {
    uint8_t read_status = 0;
    memset(camera_96717_data, 0, sizeof(camera_96717_data));
    for (auto id : Camera_ids_group_a) {
        if (id < 0 || id > 15) {
            continue;
        }
        if (id == 0) {
            for (size_t i = 0; i < 9; i++) {
                read_status =
                    m_upsensorInf->InterfaceReadRegister(id, CAMERA_REG_SER, camera_96717_buf[i], &camera_96717_data[0][i], 1);
                if (read_status != 0) {
                    camera_96717_data[0][i] = 0;
                }
            }
        } else if (id == 1) {
            for (size_t j = 0; j < 9; j++) {
                read_status =
                    m_upsensorInf->InterfaceReadRegister(id, CAMERA_REG_SER, camera_96717_buf[j], &camera_96717_data[1][j], 1);
                if (read_status != 0) {
                    camera_96717_data[1][j] = 0;
                }
            }
        }
    }
    for (auto id : Camera_ids_group_a) {
        if ((id == 0) || (id == 1)) {
            Sensor_96717_diagnose(&camera_96717_data);
        }
    }
    return 0;
}

#if 0
int CameraDiagnosis::PMIC_check(void)
{
    uint8_t X8B_id = 0;
    uint8_t n,m;
    uint8_t read_status = 0;
	
    for(auto id:Camera_ids_group)
    {
	 if(id < 0 || id > 15) continue;
	 split(id,X8B_id);
	 if(X8B_id == 0)
	 {  
            for(size_t i=0;i<5;i++) 
            {
                 read_status = PMIC_x8b_Read(PMIC_buf[i],&PMIC_data[0][i]);
                 if(read_status!=0)
                 {
                    PMIC_data[0][i] = 0;
                 }
            }
         }
         else if(X8B_id ==1 )
         {
            for(size_t j=0;j<5;j++) 
            {
                 read_status = PMIC_x8b_Read(PMIC_buf[j],&PMIC_data[1][j]);
                 if(read_status!=0)
                 {
                    PMIC_data[1][j] = 0;
                 }
            }
			
         }
    }
    PMIC_diagnose(&PMIC_data);
    return 0;
}
#endif

int CameraDiagnosis::Max96712_Safety_Setting(void) {
    uint8_t reg_temp;
    uint8_t new_reg_temp;

    for (auto id : Camera_ids_group)  //four group  = four max96712
    {
        if (id < 0 || id > 15) {
            continue;
        }

        for (size_t i = 0; i < 7; i++)  //each group has 7 regs that need to set
        {
            m_upsensorInf->InterfaceReadRegister(id, CAMERA_REG_DESER, Max96712_setting[i].reg, &reg_temp, 1);
            new_reg_temp = reg_temp;
            for (size_t a = Max96712_setting[i].start_bit; a <= Max96712_setting[i].end_bit; a++) {
                if (Max96712_setting[i].w_r_flag) {
                    setbit(new_reg_temp, a);
                } else {
                    clrbit(new_reg_temp, a);
                }
            }
            LOG_INFO("name %s     reg %02x     old_value %02x  new_value %02x \n", Max96712_setting[i].Reg_Name,
                     Max96712_setting[i].reg, reg_temp, new_reg_temp);
            m_upsensorInf->InterfaceWriteRegister(id, CAMERA_REG_DESER, Max96712_setting[i].reg, &new_reg_temp, 1);
        }
    }
    return 0;
}

void CameraDiagnosis::Max96712_Safety_Readout() {
    uint8_t flag = 0;
    uint8_t lock_flag = 0;
    uint8_t interrupt_flag = 0;
    uint8_t deser_num = 0;
    uint8_t deser_function_safety = 0;  // Self_check_DTC gmsl2 lock
    uint8_t deser_function_diagnosis = 0;

    for (auto id : Camera_ids_group)  //four group  = four max96712
    {
        if (id < 0 || id > 15) {
            continue;
        }
        split(id, deser_num);
        for (uint8_t i = 0; i < 4; i++)  //4 channel
        {
            flag = 0;
            lock_flag = 0;
            interrupt_flag = 0;
            for (uint8_t k = 0; k < 5; k++)  // 1 channel has 5 regs
            {
                if ((Max96712_detect_Reg[k + 5 * i].reg == 0x400) || ((deser_num == 2) && (k == 4))) {
                    deser_function_safety = 0x00;
                } else {
                    m_upsensorInf->InterfaceReadRegister(id, CAMERA_REG_DESER, Max96712_detect_Reg[k + 5 * i].reg, &deser_function_safety, 1);
                }
                if (deser_function_safety & (0x1 << Max96712_detect_Reg[k + 5 * i].start_bit)) {
                    LOG_WARN("function_safety_trace error value:%d bit:%d\n", deser_function_safety,
                             Max96712_detect_Reg[k + 5 * i].start_bit);
                    flag = 1;
                }
            }
            if (get_interrupt_flag) {
                for (uint8_t j = 0; j < 3; j++)  // 1 channel has 3 regs
                {
                    m_upsensorInf->InterfaceReadRegister(id, CAMERA_REG_DESER, Max96712_Interrupt_detect_Reg[j + 3 * i].reg, &deser_function_safety, 1);
                    if (deser_function_safety & (0x1 << Max96712_Interrupt_detect_Reg[j + 3 * i].start_bit)) {
                        LOG_WARN("function_safety_interrupt_trace error value:%d bit:%d\n", deser_function_safety,
                                 Max96712_Interrupt_detect_Reg[j + 3 * i].start_bit);
                        interrupt_flag = 1;
                    }
                }
            }
            /****** lock part start ******/
            if (Max96712_Link_Lock_Reg[4 * deser_num + i]) {
                m_upsensorInf->InterfaceReadRegister(id, CAMERA_REG_DESER, Max96712_Link_Lock_Reg[4 * deser_num + i], &deser_function_safety, 1);
                if ((deser_function_safety & 0x8) != 0x8) {
                    if ((id == 0) && (i == 3)) {
                        LOG_DBG("====>The fourth channel in group A is empty ! =======>\n");
                    } else {
                        LOG_WARN("function_safety_lock_trace error value:%d\n", deser_function_safety);
                        lock_flag = 1;
                    }
                }
            }
            /****** lock part end ******/

            if (flag || interrupt_flag || lock_flag) {
                LOG_WARN("flag:%d interrupt_flag:%d lock_flag:%d deser_num:%d i:%d\n", flag, interrupt_flag, lock_flag,
                         deser_num, i);
                if (deser_num == 0) {
                    switch (i) {
                        case 0:
                            cameraFaultReport.MAX96712_1_Error_Status |= 0x1 << 1;
                            break;
                        case 1:
                            cameraFaultReport.MAX96712_1_Error_Status |= 0x1 << 2;
                            break;
                        case 2:
                            cameraFaultReport.MAX96712_1_Error_Status |= 0x1 << 0;
                            break;
                        default:
                            break;
                    }
                } else if (deser_num == 1) {
                    switch (i) {
                        case 0:
                            cameraFaultReport.MAX96712_2_Error_Status |= 0x1 << 2;
                            break;
                        case 1:
                            cameraFaultReport.MAX96712_2_Error_Status |= 0x1 << 3;
                            break;
                        case 2:
                            cameraFaultReport.MAX96712_2_Error_Status |= 0x1 << 0;
                            break;
                        case 3:
                            cameraFaultReport.MAX96712_2_Error_Status |= 0x1 << 1;
                            break;
                        default:
                            break;
                    }
                } else if (deser_num == 2) {
                    switch (i) {
                        case 0:
                            cameraFaultReport.MAX96712_3_Error_Status |= 0x1 << 2;
                            break;
                        case 1:
                            cameraFaultReport.MAX96712_3_Error_Status |= 0x1 << 3;
                            break;
                        case 2:
                            cameraFaultReport.MAX96712_3_Error_Status |= 0x1 << 0;
                            break;
                        case 3:
                            cameraFaultReport.MAX96712_3_Error_Status |= 0x1 << 1;
                            break;
                        default:
                            break;
                    }
                } else if (deser_num == 3) {
                    cameraFaultReport.MAX96712_4_Error_Status = 0;
                }
            } else {
                if (deser_num == 0) {
                    switch (i) {
                        case 0:
                            cameraFaultReport.MAX96712_1_Error_Status &= ~(0x1 << 1);
                            break;
                        case 1:
                            cameraFaultReport.MAX96712_1_Error_Status &= ~(0x1 << 2);
                            break;
                        case 2:
                            cameraFaultReport.MAX96712_1_Error_Status &= ~(0x1 << 0);
                            break;
                        default:
                            break;
                    }
                } else if (deser_num == 1) {
                    switch (i) {
                        case 0:
                            cameraFaultReport.MAX96712_2_Error_Status &= ~(0x1 << 2);
                            break;
                        case 1:
                            cameraFaultReport.MAX96712_2_Error_Status &= ~(0x1 << 3);
                            break;
                        case 2:
                            cameraFaultReport.MAX96712_2_Error_Status &= ~(0x1 << 0);
                            break;
                        case 3:
                            cameraFaultReport.MAX96712_2_Error_Status &= ~(0x1 << 1);
                            break;
                        default:
                            break;
                    }
                } else if (deser_num == 2) {
                    switch (i) {
                        case 0:
                            cameraFaultReport.MAX96712_3_Error_Status &= ~(0x1 << 2);
                            break;
                        case 1:
                            cameraFaultReport.MAX96712_3_Error_Status &= ~(0x1 << 3);
                            break;
                        case 2:
                            cameraFaultReport.MAX96712_3_Error_Status &= ~(0x1 << 0);
                            break;
                        case 3:
                            cameraFaultReport.MAX96712_3_Error_Status &= ~(0x1 << 1);
                            break;
                        default:
                            break;
                    }
                } else if (deser_num == 3) {
                    cameraFaultReport.MAX96712_4_Error_Status = 0;
                }
            }
        }
        deser_function_diagnosis = 0x00;
        m_upsensorInf->InterfaceReadRegister(id, CAMERA_REG_DESER, 0x0D, &deser_function_diagnosis, 1);
        if (Deserdes_ID[deser_num] != deser_function_diagnosis) {
            DTC_ERROR.Camera_Number[deser_num].CAM_DTC_ERR[0].DTC_ShowNum = reg_dtc_map[0][1 + deser_num];
            DTC_ERROR.Camera_Number[deser_num].CAM_DTC_ERR[0].DTC_Error_Status = 0x01;
        }
    }

    Clear_reg();  //clean the reg by read these reg after diag
    get_interrupt_flag = 0;
}

/******************************** Functional Safety Part Max96712 End ******************************************/

int CameraDiagnosis::Max20087_Diagnosis(void) {
    uint8_t MASK = 0x00;
    uint8_t max20087_num;
    uint8_t campwr_reg[4][10] = {0};
    for (auto id : Camera_ids_group) {
        if (id < 0 || id > 15) {
            continue;
        }
        split(id, max20087_num);
        for (uint8_t a = 0; a < 10; a++) {
            m_upsensorInf->InterfaceReadRegister(id, CAMERA_REG_CAMPWR, MASK + a, &campwr_reg[max20087_num][a], 1);
        }
    }
    Max20087_diagnose(campwr_reg);

    return 0;
}

void* CameraDiagnosis::diagnose_ThreadFunc() {
    LOG_INFO("diagnose_thread id:%ld\n", pthread_self());
    _CameraFaultReport CustomFaultReport = {0};  //customer add their changes here
    // std::this_thread::sleep_for(std::chrono::seconds(3));
    for (;;) {
        DTCError_Init(&DTC_ERROR);
        Max20087_Diagnosis();
        Sensor_Safety_check();
        Sensor_96717_check();
        //PMIC_check();
        Max96712_Safety_Readout();
        Serializer_Read();

        CustomFaultReport = safety_error_status_interface();
        printf("CustomFaultReport->Sensor_X8B_0_Error_Status %04x \n", CustomFaultReport.Sensor_X8B_0_Error_Status);
        printf("CustomFaultReport->Sensor_X8B_1_Error_Status %04x \n", CustomFaultReport.Sensor_X8B_1_Error_Status);
        printf("CustomFaultReport->MAX96712_1_Error_Status %04x \n", CustomFaultReport.MAX96712_1_Error_Status);
        printf("CustomFaultReport->MAX96712_2_Error_Status %04x \n", CustomFaultReport.MAX96712_2_Error_Status);
        printf("CustomFaultReport->MAX96712_3_Error_Status %04x \n", CustomFaultReport.MAX96712_3_Error_Status);
        // printf("CustomFaultReport->MAX96712_4_Error_Status %04x \n", CustomFaultReport.MAX96712_4_Error_Status);
        printf("CustomFaultReport->MAX20087_1_OUV_Error_Status %04x \n", CustomFaultReport.MAX20087_1_OUV_Error_Status);
        printf("CustomFaultReport->MAX20087_2_OUV_Error_Status %04x \n", CustomFaultReport.MAX20087_2_OUV_Error_Status);
        printf("CustomFaultReport->MAX20087_3_OUV_Error_Status %04x \n", CustomFaultReport.MAX20087_3_OUV_Error_Status);
        // printf("CustomFaultReport->MAX20087_4_OUV_Error_Status %04x \n", CustomFaultReport.MAX20087_4_OUV_Error_Status);

        if (cameraFaultReport.MAX96712_1_Error_Status) {
            CameraDiagnosisDesFault(4580, cameraFaultReport.MAX96712_1_Error_Status);
        }

        if (cameraFaultReport.MAX96712_2_Error_Status) {
            CameraDiagnosisDesFault(4590, cameraFaultReport.MAX96712_2_Error_Status);
        }

        if (cameraFaultReport.MAX96712_3_Error_Status) {
            CameraDiagnosisDesFault(4600, cameraFaultReport.MAX96712_3_Error_Status);
        }

        if (cameraFaultReport.MAX20087_1_OUV_Error_Status) {
            CameraDiagnosisPowerFault(1, cameraFaultReport.MAX20087_1_OUV_Error_Status);
        }

        if (cameraFaultReport.MAX20087_2_OUV_Error_Status) {
            CameraDiagnosisPowerFault(2, cameraFaultReport.MAX20087_2_OUV_Error_Status);
        }

        if (cameraFaultReport.MAX20087_3_OUV_Error_Status) {
            CameraDiagnosisPowerFault(3, cameraFaultReport.MAX20087_3_OUV_Error_Status);
        }

        for (u_int8_t i = 0; i < camera_num; i++)  //12 camera
        {
            for (u_int8_t j = 0; j < dtc_num; j++)  //6*DTC
            {
                printf("Camera_Number[%d] | DTC_ERR[%d].DTC_ShowNum = %s | DTC_Error_Status = %d \n", i, j,
                       DTC_ERROR.Camera_Number[i].CAM_DTC_ERR[j].DTC_ShowNum,
                       DTC_ERROR.Camera_Number[i].CAM_DTC_ERR[j].DTC_Error_Status);
            }
            printf("--------------------------------------------------------------------------------\n");
        }

        for (u_int8_t i = 0; i < 2; i++)  //5 ser & deser
        {
            for (u_int8_t j = 0; j < 1; j++)  //6*DTC
            {
                printf("Serder_Number[%d] | DTC_ERR[%d].DTC_ShowNum = %s | DTC_Error_Status = %d \n", i, j,
                       DTC_ERROR.Serder_Number[i].CAM_DTC_ERR[j].DTC_ShowNum,
                       DTC_ERROR.Serder_Number[i].CAM_DTC_ERR[j].DTC_Error_Status);
            }
            printf("--------------------------------------------------------------------------------\n");
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    return NULL;
}

void CameraDiagnosis::CameraDiagnosisReportFault(uint32_t cameraFaultId, uint8_t cameraFaultObj) {
    uint8_t cameraFaultStatus = 1;
    LOG_WARN("camera diagnose id :%d obj:%d\n", cameraFaultId, cameraFaultObj);
    SendFault_t cameraFault(cameraFaultId, cameraFaultObj, cameraFaultStatus);
    CameraPhmClient::getInstance()->CameraReportFault(cameraFault);
}

void CameraDiagnosis::CameraDiagnosisDesFault(uint32_t cameraFaultId, uint16_t error_status) {
    uint8_t cameraFaultObj = 0;
    if (error_status & (0x1 << 0)) {
        cameraFaultObj = 1;
        CameraDiagnosisReportFault(cameraFaultId, cameraFaultObj);
    }
    if (error_status & (0x1 << 1)) {
        cameraFaultObj = 2;
        CameraDiagnosisReportFault(cameraFaultId, cameraFaultObj);
    }
    if (error_status & (0x1 << 2)) {
        cameraFaultObj = 3;
        CameraDiagnosisReportFault(cameraFaultId, cameraFaultObj);
    }
    if (error_status & (0x1 << 3)) {
        cameraFaultObj = 4;
        CameraDiagnosisReportFault(cameraFaultId, cameraFaultObj);
    }
}

void CameraDiagnosis::CameraDiagnosisPowerFault(uint8_t powerId, uint16_t error_status) {
    uint32_t cameraFaultId = 0;
    uint8_t cameraFaultObj = 0;
    switch (powerId) {
        case 1:  // MAX20087_1
            if (error_status & (0x1 << 0)) {
                cameraFaultId = 4570;
                cameraFaultObj = 1;
                CameraDiagnosisReportFault(cameraFaultId, cameraFaultObj);
            }
            if (error_status & (0x1 << 1)) {
                cameraFaultId = 4510;
                cameraFaultObj = 1;
                CameraDiagnosisReportFault(cameraFaultId, cameraFaultObj);
            }
            if (error_status & (0x1 << 2)) {
                cameraFaultId = 4510;
                cameraFaultObj = 2;
                CameraDiagnosisReportFault(cameraFaultId, cameraFaultObj);
            }
            if (error_status & (0x1 << 3)) {
                cameraFaultId = 4510;
                cameraFaultObj = 3;
                CameraDiagnosisReportFault(cameraFaultId, cameraFaultObj);
            }
            break;
        case 2:  // MAX20087_2
            if (error_status & (0x1 << 0)) {
                cameraFaultId = 4570;
                cameraFaultObj = 2;
                CameraDiagnosisReportFault(cameraFaultId, cameraFaultObj);
            }
            if (error_status & (0x1 << 1)) {
                cameraFaultId = 4510;
                cameraFaultObj = 4;
                CameraDiagnosisReportFault(cameraFaultId, cameraFaultObj);
            }
            if (error_status & (0x1 << 2)) {
                cameraFaultId = 4510;
                cameraFaultObj = 5;
                CameraDiagnosisReportFault(cameraFaultId, cameraFaultObj);
            }
            if (error_status & (0x1 << 3)) {
                cameraFaultId = 4510;
                cameraFaultObj = 6;
                CameraDiagnosisReportFault(cameraFaultId, cameraFaultObj);
            }
            if (error_status & (0x1 << 4)) {
                cameraFaultId = 4510;
                cameraFaultObj = 7;
                CameraDiagnosisReportFault(cameraFaultId, cameraFaultObj);
            }
            break;
        case 3:  // MAX20087_3
            if (error_status & (0x1 << 0)) {
                cameraFaultId = 4570;
                cameraFaultObj = 3;
                CameraDiagnosisReportFault(cameraFaultId, cameraFaultObj);
            }
            if (error_status & (0x1 << 1)) {
                cameraFaultId = 4510;
                cameraFaultObj = 8;
                CameraDiagnosisReportFault(cameraFaultId, cameraFaultObj);
            }
            if (error_status & (0x1 << 2)) {
                cameraFaultId = 4510;
                cameraFaultObj = 9;
                CameraDiagnosisReportFault(cameraFaultId, cameraFaultObj);
            }
            if (error_status & (0x1 << 3)) {
                cameraFaultId = 4510;
                cameraFaultObj = 10;
                CameraDiagnosisReportFault(cameraFaultId, cameraFaultObj);
            }
            if (error_status & (0x1 << 4)) {
                cameraFaultId = 4510;
                cameraFaultObj = 11;
                CameraDiagnosisReportFault(cameraFaultId, cameraFaultObj);
            }
            break;

        default:
            break;
    }
}

void CameraDiagnosis::CameraDiagnosisInit(const uint32_t uSensor) {
    if (uSensor < 4) {
        Camera_ids_group_a.push_back(uSensor);
    } else if (4 <= uSensor && uSensor < 8) {
        Camera_ids_group_b.push_back(uSensor);
    } else if (8 <= uSensor && uSensor < 12) {
        Camera_ids_group_c.push_back(uSensor);
    } else if (12 <= uSensor && uSensor < 16) {
        Camera_ids_group_d.push_back(uSensor);
    }

    if (!Camera_ids_group_a.empty()) {
        Camera_ids_group[0] = *std::min_element(Camera_ids_group_a.begin(), Camera_ids_group_a.end());
    }

    if (!Camera_ids_group_b.empty()) {
        Camera_ids_group[1] = *std::min_element(Camera_ids_group_b.begin(), Camera_ids_group_b.end());
    }

    if (!Camera_ids_group_c.empty()) {
        Camera_ids_group[2] = *std::min_element(Camera_ids_group_c.begin(), Camera_ids_group_c.end());
    }

    if (!Camera_ids_group_d.empty()) {
        Camera_ids_group[3] = *std::min_element(Camera_ids_group_d.begin(), Camera_ids_group_d.end());
    }
}

void CameraDiagnosis::CameraDiagnosisStart() {
    Max96712_Safety_Setting();
    DTCError_Init(&DTC_ERROR);  //clear last dtc code
    m_upThread.reset(new std::thread(&CameraDiagnosis::diagnose_ThreadFunc, this));
}

void CameraDiagnosis::CameraDiagnosisNotify(uint32_t sensorId, uint32_t cameraFaultId) {
    uint8_t cameraFaultObj = faultIndexArray[sensorId];
    if ((cameraFaultObj > 0) && (cameraFaultObj < MAX_VAILD_SENSORS_NUM)) {
        CameraDiagnosisReportFault(cameraFaultId, cameraFaultObj);
    }
}

void CameraDiagnosis::CameraDiagnosisSetInitFlag(uint8_t flag) {
    get_interrupt_flag = flag;
}
