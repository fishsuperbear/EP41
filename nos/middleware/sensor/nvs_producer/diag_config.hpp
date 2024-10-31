#ifndef diag_config_HPP
#define diag_config_HPP


#include <stdio.h>
#include "fcntl.h"
#include "stdlib.h"
#include <poll.h>
#include <sys/select.h>
#include <sys/time.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <stdint.h>
#include <unistd.h> 
#include <string.h>
#include <thread>
#include <iostream>
#include <linux/i2c.h>
#include <linux/i2c-dev.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <netinet/in.h>
#include <arpa/inet.h>


/********************************************************************************
	PORTC                       	PORTB                      	PORTA                                               
-------------------------       -------------------------      -------------------------                                                       
| 3(3A out4)  0(3A out1) |      | 3(2B out2)  0(2B out1) |     | 3(1B out4)  0(1B out3) |                                          
|						 |      |						 |     |						|                              
| 2(3A out2)  1(3A out3) |		| 2(2B out4)  1(2B out3) |	   | 2(1B out2)  1(1B out1) |			                			
-------------------------       -------------------------      -------------------------           
*********************************************************************************/
#define Reg_MASK       0x00
#define Reg_CONFIG     0x01
#define Reg_ID         0x02
#define Reg_STAT1      0x03
#define Reg_STAT2_A    0x04
#define Reg_STAT2_B    0x05
#define Reg_ADC1       0x06
#define Reg_ADC2       0x07
#define Reg_ADC3       0x08
#define Reg_ADC4       0x09


#define dtc_num 4
#define camera_num 12
#define group_num 4

extern const char  *reg_dtc_map[dtc_num][camera_num+1];

#define setbit(x,y)  x|=(1<<y)
#define clrbit(x,y)  x&=~(1<<y)
#define split(in,out) out=(in<4) ? 0:((4 <= in && in < 8)? 1:((8 <= in && in < 12) ? 2:((12 <= in && in < 16)? 3:3 )))

#define MAX96712_DEV_ID         0xA0u
#define MAX96722_DEV_ID         0xA1u
#define MAX96724_DEV_ID         0xA2u
#define MAX96724F_DEV_ID        0xA3u

/******************************** Diagnosis Start ******************************************/
enum Reg_Name{
	MASK,
	CONFIG,
	ID,
	STAT1,
	STAT2_A,
	STAT2_B,
	ADC1,
	ADC2,       
	ADC3,
	ADC4
};

struct _DTC_ERROR_Content{
	const char *DTC_ShowNum;
	uint8_t DTC_Error_Status;
};

struct  _DTC_ERROR_Type{
	_DTC_ERROR_Content CAM_DTC_ERR[dtc_num];
};

struct  _DTC_ERROR{
    _DTC_ERROR_Type Serder_Number[2];
	_DTC_ERROR_Type Camera_Number[camera_num+1];
};

extern struct  _DTC_ERROR  DTC_ERROR;

enum CameraRelatedDiagFault
{
    NoFault                     = 0,
    I2cCommunicationFailure     = 1,
    OpenCircuit                 = 2,
    Short_to_ground             = 3,
    Short_to_battery            = 4,
};

struct CommunicationComponent
{
    CameraRelatedDiagFault Max96712_1;
    CameraRelatedDiagFault Max96722_2;
    CameraRelatedDiagFault Max96722_3;
    CameraRelatedDiagFault Max96717;
    CameraRelatedDiagFault Max96781;
};

struct Camera
{
    CameraRelatedDiagFault CAM_FR_120;
    CameraRelatedDiagFault CAM_FR_30;
    CameraRelatedDiagFault CAM_R;
    CameraRelatedDiagFault CAM_LF;
    CameraRelatedDiagFault CAM_LR;
    CameraRelatedDiagFault CAM_RF;
    CameraRelatedDiagFault CAM_RR;
    CameraRelatedDiagFault AVM_F_CAM;
    CameraRelatedDiagFault AVM_L_CAM;
    CameraRelatedDiagFault AVM_R_CAM;
    CameraRelatedDiagFault AVM_RR_CAM;
};

/******************************** Diagnosis End ******************************************/


/******************************** Functional Safety Part Max96712 Start ******************************************/

struct Max96712_Detect_Struct{
    char const   *Reg_Name;
    uint16_t     reg;
    uint8_t      start_bit;
    uint8_t      end_bit;
    char         w_r_flag;
};

/**************Safety ************/

struct _CameraFaultReport
{
   uint16_t Sensor_X8B_0_Error_Status;
   uint16_t Sensor_X8B_1_Error_Status;
   uint16_t MAX96712_1_Error_Status;
   uint16_t MAX96712_2_Error_Status;
   uint16_t MAX96712_3_Error_Status;
   uint16_t MAX96712_4_Error_Status;
   uint16_t MAX20087_1_OUV_Error_Status;
   uint16_t MAX20087_2_OUV_Error_Status;
   uint16_t MAX20087_3_OUV_Error_Status;
   uint16_t MAX20087_4_OUV_Error_Status;
};

extern struct _CameraFaultReport cameraFaultReport;
/*******************************/

/******************************** Functional Safety Part Max96712 End   ******************************************/
extern int Sensor_diagnose(uint8_t (*camera_data)[2][20]);
extern int Sensor_96717_diagnose(uint8_t (*camera_96717_data)[2][20]);
//extern int PMIC_diagnose(uint8_t (*PMIC_data)[2][20]);
extern int Max20087_diagnose(uint8_t (*reg)[10]);
extern int Max20087_Track_Diag_Func(void);
extern int Max20087_Pre_Read(void);
extern int Serializer_Read(void);
#if 0
extern int PMIC_x8b_Read(uint16_t PMIC_adress,uint8_t * PMIC_temp);
#endif
//extern int Get_Electrical_Error(uint8_t max20087_num,struct _DTC_ERROR *dtc_error, uint8_t reg[]);
extern int Diagnosis_Debug(struct _DTC_ERROR *dtc_error);
extern int DTCError_Init(struct _DTC_ERROR *dtc_error);
//extern int Get_Electrical_Error(struct _DTC_ERROR *dtc_error, struct _CameraFaultReport *camera_fault_error, vector<uint8_t> *oc_vet,  vector<uint8_t> *sg_vet,  vector<uint8_t> *ov_vet, vector<uint8_t> *uv_vet);

//extern const char  *reg_dtc_map[dtc_num][camera_num+1];
extern uint8_t  Deserdes_ID[3];
extern uint16_t mcam_i2c_reg[2][12];

extern uint16_t Max96712_Clear_Reg[7];
extern Max96712_Detect_Struct Max96712_detect_Reg[28];
extern Max96712_Detect_Struct Max96712_setting[7];

extern uint16_t Max96712_Interrupt_Clear_Reg[8];
extern Max96712_Detect_Struct Max96712_Interrupt_detect_Reg[12];
extern uint16_t Max96712_Link_Lock_Reg[16];

extern _CameraFaultReport safety_error_status_interface();

#endif
