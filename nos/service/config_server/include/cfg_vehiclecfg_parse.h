/*
 * @Author: Z04975 liguoqiang@hozonauto.com
 * @Date: 2023-11-08 17:31:49
 * @LastEditors: Z04975 liguoqiang@hozonauto.com
 * @LastEditTime: 2023-12-25 11:37:57
 * @FilePath: /nos/service/config_server/include/cfg_vehiclecfg_parse.h
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */

/*
 * Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
 * Module: per
 * Description: 整车配置字解析
 * Created on: Feb 7, 2023
 *
 */
#ifndef SERVICE_CONFIG_SERVER_INCLUDE_CFG_VEHICLECFG_H_
#define SERVICE_CONFIG_SERVER_INCLUDE_CFG_VEHICLECFG_H_
#include <dirent.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <ctime>
#include <iostream>
#include <iterator>
#include <map>
#include <shared_mutex>
#include <sstream>
#include <string>
#include <typeinfo>
#include <vector>

#include "include/cfg_data_def.h"
#include "include/cfg_logger.h"
#include "include/cfg_server_data_def.h"
#include "include/cfg_utils.h"
#include "phm/include/phm_client.h"
#include "service/config_server/include/cfg_vehiclecfg_update.h"
namespace hozon {
namespace netaos {
namespace cfg {
const std::vector<std::pair<std::string, uint8_t>> vehiclecfgkeyvec = {
    {"platform", 4},                                         // 平台
    {"brand", 4},                                            // 品牌
    {"vehicleDesignCode", 8},                                // 整车设计代码
    {"salesMarket", 8},                                      // 销售市场
    {"frontMotorPower", 5},                                  // 前电机功率
    {"frontMotorType", 3},                                   // 前电机类型
    {"frontmotorManufacturer", 8},                           // 前电机厂家
    {"frontGearRatio", 8},                                   // (前)GearRatio(速比)
    {"batteriesSeriesNumber", 8},                            // 电池串联数（0~255）(需配置HVM)
    {"rearMotorPower", 5},                                   // 后电机功率
    {"rearMotorType", 3},                                    // 后电机类型
    {"batteryLife", 8},                                      // 电池续航
    {"batteryCapacity", 8},                                  // 电池容量(KW)
    {"batteryType", 3},                                      // 电池类型
    {"steeringWheelPosition", 1},                            // 方向盘位置
    {"maxSpeed", 4},                                         // 最高车速
    {"batteryManufacturer", 8},                              // 电池厂家
    {"shiftForm", 3},                                        // 换挡形式
    {"powerBatteryCoolMode", 1},                             // 动力电池冷却方式
    {"chargerCoolMode", 1},                                  // 充电机冷却方式
    {"onBoardChargerPower", 3},                              // 车载充电器功率
    {"batteryTempSensorsNumber", 8},                         // 电池温度传感器数量（0~127）(需配置HVM)
    {"FwvModule", 1},                                        // FWV模块
    {"threeWayValveTWV4", 1},                                // 三通阀TWV4
    {"threeWayValveTWV3", 1},                                // 三通阀TWV3
    {"threeWayValveTWV2", 1},                                // 三通阀TWV2
    {"onepedalFunc", 1},                                     // onepedal 功能(电子油门踏板ONE PEDAL功能模式功能)
    {"externalDischarge", 1},                                // 对外放电V2L/V2V（车对负载，车对车放电）
    {"hvm", 1},                                              // HVM（高压采集模块）
    {"evcc", 1},                                             // EVCC
    {"drivingPowerForm", 3},                                 // 驱动动力形式
    {"ags", 1},                                              // AGS主动进气格栅
    {"soundAlgorithm", 3},                                   // 音效算法
    {"ethernetMode", 1},                                     // 以太网模式
    {"moduleSupplier", 8},                                   // 模组供应商(需配置HVM)
    {"driverAirbagSquib", 1},                                // driver airbag squib 驾驶员气囊
    {"passengerAirbagSquib", 1},                             // passenger airbag squib 副驾驶员气囊
    {"leftFrontPretensionerSquib", 1},                       // left front side pretensioner squib 前排左侧安全带预紧器
    {"rightFrontPretensionerSquib", 1},                      // right front side pretensioner squib 前排右侧安全带预紧器
    {"leftFrontAirbagSquib", 1},                             // left front side airbag squib 前排左侧侧面气囊
    {"rightFrontAirbagSquib", 1},                            // right front side airbag squib 前排右侧侧面气囊
    {"leftcurtainAirbagSquib", 1},                           // left side curtain airbag squib 左侧安全气帘
    {"rightcurtainAirbagSquib", 1},                          // right side  curtain airbag squib 右侧安全气帘
    {"reserved_5", 4},                                       // 预留
    {"rightFrontPreloadAnchor", 1},                          // 前排右侧锚式预紧
    {"leftFrontPreloadAnchor", 1},                           // 前排左侧锚式预紧
    {"driveRearPretensionerSquib", 1},                       // driver rear pretensioner squib 驾驶员侧后排安全带卷收器预紧
    {"passengerPretensionerSquib", 1},                       // passenger rear pretensioner squib 副驾驶员侧后排安全带卷收器预紧
    {"reserved_6", 1},                                       // 预留
    {"driverSafetyBeltLockCatch", 1},                        // driver Safety Belt lock catch 驾驶员安全带锁扣开关
    {"rightPressureSensor", 1},                              // right side  pressure sensor右侧压力传感器
    {"leftPressureSensor", 1},                               // left side  pressure sensor左侧压力传感器
    {"rightCrashSensor", 1},                                 // right side crash sensor 右中柱侧面碰撞传感器
    {"leftCrashSensor", 1},                                  // left side crash sensor 左中柱侧面碰撞传感器
    {"rightFrontCrashSensor", 1},                            // right front side crash sensor 右前正面碰撞传感器
    {"leftFrontCrashSensor", 1},                             // left front side crash sensor 左前正面碰撞传感器
    {"passengerSafetyBeltLockCatch", 1},                     // passenger safety belt lock catch 副驾驶员安全带锁扣开关
    {"passengerDetect", 1},                                  // passenger detect 副驾乘员检测
    {"leftRearSafetyBeltLockCatch", 1},                      // left rear safety belt lock catch 左后排全带锁扣开关
    {"leftRearSafetyBeltLockCatchPassengerDetect", 1},       // left rear safety belt lock catch +passenger detect 左后排安全带锁扣+乘员检测
    {"middleRearSafetyBeltLockCatch", 1},                    // middle rear safety belt lock catch 后中座位全带锁扣开关
    {"middleRearSafetyBeltLockCatchPassengerDetect", 1},     // middle rear safety belt lock catch +passenger detect后中座位安全带锁扣+乘员检测
    {"rightRearSafetyBeltLockCatch", 1},                     // right rear safety belt lock catch 右后排全带锁扣开关
    {"rightRearSafetyBeltLockCatchPassengerDetect", 1},      // right rear safety belt lock catch +passenger detect右后排安全带锁扣+乘员检测
    {"reserved_7", 8},                                       // 预留
    {"driverSeatAdjustType", 2},                             // 驾驶员座椅调节类型
    {"passengerSeatMassageFunc", 1},                         // 副驾驶员座椅按摩功能（SMCU_R模块）
    {"driverSeatMassageFunc", 1},                            // 驾驶员座椅按摩功能（SMCU_L模块）
    {"switchWiperSignalType", 1},                            // 组合开关雨刮信号类型
    {"autoWiper", 1},                                        // 自动雨刮
    {"parkUnlock", 2},                                       // 驻车解锁
    {"passengerSeatVentilationFunc", 1},                     // 副驾驶员座椅通风功能
    {"passengerSeatHeatingFunc", 1},                         // 副驾驶员座椅加热功能
    {"passengerSeatMemoryFunc", 1},                          // 副驾驶员座椅记忆功能
    {"vehicleSpeedLock", 2},                                 // VehicleSpeedLock（车速上锁）
    {"doorHandleAntennasCnt", 2},                            // 门把手天线数量
    {"doorOpenDoubleFlashRemind", 1},                        // 开门双闪提醒(TBD)
    {"rearSeatVentilation", 1},                              // 后排座椅通风
    {"skylightType", 2},                                     // 天窗类型
    {"turnSignalDiag", 1},                                   // 转向灯诊断
    {"searchSettings", 1},                                   // 寻车设置
    {"keylessEntryFunc", 2},                                 // 无钥匙进入功能
    {"rainCloseWindow", 1},                                  // 下雨关窗
    {"reserved_1", 1},                                       // reserved
    {"eleExteriorRearviewMirrorMemAutoReverseFlipDown", 1},  // 电动外后视镜记忆+倒车自动下翻
    {"eleHeatingFuncORVM", 1},                               // 外后视镜带电加热功能
    {"eleLensAdjustFuncORVM", 1},                            // 电动外后视镜镜片调节
    {"autoExteriorFoldFuncORVM", 1},                         // 自动外后视镜折叠
    {"antiTheftVCU", 1},                                     // VCU防盗（IMMO认证）
    {"escl", 1},                                             // ESCL
    {"brakeLampDiag", 1},                                    // 制动灯诊断
    {"doorOpeningMode", 3},                                  // 车门打开方式
    {"passengerSeatAdjustType", 2},                          // 副驾座椅及腿托调节类型
    {"reserved_8", 1},                                       // 预留
    {"headlampSwitchSigType", 1},                            // 前照灯开关信号类型（位置+AUTO+近光）
    {"turnSigDirectDrive", 1},                               // 转向灯直驱
    {"driverSeatVentilationFunc", 1},                        // 主驾座椅通风功能
    {"driverSeatHeatingFunc", 1},                            // 主驾座椅加热功能
    {"rearDoorChildSafetyLock", 1},                          // 后车门儿童安全锁
    {"windowGlassPushupFunc", 2},                            // 车窗玻璃一键升功能
    {"relatedRarklightDRL", 1},                              // DRL related parklight
    {"runWaterWelcomeLamp", 1},                              // 流水迎宾灯
    {"autoReverseRearWiper", 1},                             // 倒车自动后雨刮
    {"multifuncSteeringWheel", 1},                           // 多功能方向盘
    {"windowSwitch", 1},                                     // 车窗开关
    {"antiPinch", 2},                                        // anti-pinch（车窗升降防夹功能）
    {"drivingMemory", 1},                                    // 主驾记忆
    {"secondRowSeatHeat", 1},                                // 第二排座椅加热
    {"oms", 1},                                              // 后排儿童照看/生物遗留探测/全舱监控系统（OMS）
    {"centralLockingCfg", 1},                                // 中控锁配置
    {"frontWheelSPVal", 4},                                  // Standard Pressure Value For FrontWheel 前轮标准胎压
    {"rearWheelSPVal", 4},                                   // Standard Pressure Value For RearWheel 后轮标准胎压
    {"reserved_4", 1},                                       // 预留
    {"fourdoorglassCfg", 2},                                 // 四门玻璃配置DCU
    {"bam", 1},                                              // 蓝牙鉴权模块
    {"reserved_2", 2},                                       // 预留
    {"windowLowerRC", 2},                                    // 遥控降窗
    {"reserved_3", 2},                                       // 预留
    {"winterMode", 1},                                       // WinterMode 冬季模式（冬季换胎使用）
    {"thresholdCfgTPMA", 2},                                 // 胎压高温报警阀值配置
    {"ecuModule", 1},                                        // EDU模块(电子驱动单元)
    {"ks", 1},                                               // 感应电动行李箱（KS）
    {"plg", 1},                                              // 电动尾门（PLG)
    {"tyreHighPressure", 4},                                 // TyreHighPressure For RearWheel 高压胎压标准
    {"doorHandleType", 2},                                   // 门把手类型
    {"frontRightBackrestHeating", 1},                        // 副驾靠背加热
    {"driverRightBackrestHeating", 1},                       // 主驾靠背加热
    {"airConditionHeatPump", 1},                             // 空调热泵(自研热泵)
    {"coofanpwmcon", 1},                                     // Coofanpwmcon(冷却风扇)
    {"telecontrol", 1},                                      // Telecontrol（远程控制）
    {"PM25detsys", 1},                                       // PM25detsys（PM2.5检测系统）
    {"airclesys", 1},                                        // Airclesys（空气净化系统负离子发生器）
    {"airquasys", 1},                                        // Airquasys（空气质量系统AQS）
    {"acconsystype", 2},                                     // Acconsystype（空调控制系统类型）
    {"fillerCapOpenForm", 1},                                // 加油口盖开启形式
    {"twv1", 1},                                             // 可调三通阀1（TWV1）
    {"fragranceSystem", 1},                                  // 香氛系统
    {"eacp", 1},                                             // EACP（电子压缩机）
    {"ptc", 1},                                              // PTC
    {"heatPumpAC", 1},                                       // 空调热泵(华为热泵）
    {"atmosphereLamp", 1},                                   // 氛围灯（功能）
    {"ctuMode", 1},                                          // CTU模式(华为热泵）
    {"reserved_10", 8},                                      // 预留
    {"reserved_9", 1},                                       // 预留
    {"reserved_11", 1},                                      // 预留
    {"frontFogLamp", 1},                                     // 前雾灯
    {"directDriveAngleLampBDCS", 1},                         // BDCS直驱角灯
    {"directDriveDaytimeRunnLampBDCS", 1},                   // BDCS直驱日间行车灯
    {"directDrivePositionLampBDCS", 1},                      // BDCS直驱位置灯
    {"directDriveHighLampBDCS", 1},                          // BDCS直驱远光灯
    {"directDriveLowBeamLampBDCS", 1},                       // BDCS直驱近光灯
    {"reserved_12", 1},                                      // 预留
    {"secondRowBackrestElectricAdjust", 1},                  // 二排靠背电动调节
    {"frontElectricAirOutlet", 1},                           // 前排电动出风口
    {"temperatureAdjustAreaAC", 1},                          // 空调温度可调区域
    {"regulationModeAC", 1},                                 // 空调系统调节方式
    {"reserved_15", 1},                                      // 预留
    {"exv2", 1},                                             // 冷凝器电子膨胀阀(EXV2)
    {"exv1", 1},                                             // 冷凝器电子膨胀阀(EXV1)
    {"SecRowSeatBeltUnfastenedReminder", 1},                 // 二排安全带未系提醒
    {"DriverSeatLumbarSupport", 2},                          // 主驾座椅腰托
    {"NumberOfMicrophones", 3},                              // 麦克风数量（Reserved）
    {"IBS", 1},                                              // IBS（蓄电池传感器）
    {"DMS", 1},                                              // 驾驶员检测摄像头(DMS)/人脸识别/疲劳驾驶
    {"sentinelMode", 1},                                     // 哨兵模式(24h停车监测）
    {"psd", 1},                                              // PSD（副驾显示屏
    {"speaker", 3},                                          // speaker（扬声器数量）
    {"brandAudio", 1},                                       // BrandAudio（品牌音响）
    {"faderBalance", 1},                                     // Fader and Balance
    {"SoundEffect3D", 1},                                    // 3D Sound Effect （3D音效）
    {"curbWeight", 3},                                       // 整备质量
    {"RadioCfg", 3},                                         // 收音机配置
    {"wpc", 1},                                              // WPC(手机无线充电)
    {"NetaModeUIDisplay", 1},                                // 哪吒音响模式UI显示
    {"RCPRearCtrlPanel", 1},                                 // RCP后控制面板
    {"reserved_13", 2},                                      // 预留
    {"VLANtechnology", 1},                                   // VLAN技术
    {"DCLCPolePullLaneChanging", 1},                         // DCLC拔杆变道/驾驶员确认变道
    {"ELKAEmergencyLaneKeeping", 1},                         // ELKA紧急车道保持
    {"narrowDoorCollisionWarning", 1},                       // 窄道开门碰撞提示
    {"frontVehicleStartReminder", 1},                        // 前车起步提示
    {"hns2Module", 1},                                       // HNS2模块
    {"hns1Module", 1},                                       // HNS1模块
    {"adasDomainCtrl", 2},                                   // ADAS域控制器
    {"highPrecisionPositionSys", 1},                         // 高精定位系统
    {"arHud", 1},                                            // AR-HUD
    {"dvr", 2},                                              // DVR
    {"holographicImage", 1},                                 // 全息影像（透明底盘）
    {"mod", 1},                                              // MOD低速动态物体识别
    {"lidar", 2},                                            // 激光雷达
    {"parkRadar", 3},                                        // 泊车雷达
    {"factoryMode", 1},                                      // 出厂模式
    {"autoSpeedLimitRegulation", 1},                         // 自动限速调节
    {"constantSpeedCruiseCategory", 2},                      // 定速巡航类别
    {"hba", 1},                                              // HBA(智能远近光灯)
    {"tsr", 1},                                              // TSR(交通标志识别与提醒)
    {"rcw", 1},                                              // RCW(后向碰撞预警)
    {"fcw", 1},                                              // FCW(前向碰撞预警)
    {"dow", 1},                                              // DOW(开门预警)
    {"ldw", 1},                                              // LDW(车道偏离预警)
    {"ntp", 1},                                              // NTP(记忆泊车)
    {"tba", 1},                                              // TBA(原路返回辅助)
    {"nms", 1},                                              // NMS（智能召唤）
    {"nvp", 1},                                              // NVP(代客泊车)
    {"apa", 1},                                              // APA（全自动泊车）
    {"otrp", 1},                                             // 一键遥控泊车
    {"fapa", 1},                                             // FAPA(融合泊车辅助)
    {"ADASDomainCtrl", 5},                                   // ADAS域控制器(续)
    {"reserved_14", 1},                                      // 预留
    {"AngularRadar", 2},                                     // 角雷达
    {"tja", 1},                                              // TJA(交通拥堵辅助)
    {"lka", 1},                                              // LKA(车道保持）
    {"advancedAutopilotFunc", 1},                            // 高阶自动驾驶功能
    {"uandaNCP", 1},                                         // 城市自动导航辅助驾驶（城市领航NCP）
    {"hsanadNNP", 1},                                        // 高速自动导航辅助驾驶NNP（哪吒智能领航辅助NNP）
    {"hwa", 1},                                              // HWA(高速变道辅助)
    {"ica", 1},                                              // ICA(智能巡航辅助)
    {"directDriveLogoLampBDCS", 1},                          // BDCS直驱Logo灯
    {"parallelAuxiliaryLCA", 1},                             // LCA并线辅助（含BSD功能）
    {"aeb", 1},                                              // AEB(自动紧急刹车)
    {"intelligentVehicleSearch", 1},                         // 智能寻车
    {"rcta", 1},                                             // RCTA(后向目标横穿警告)
    {"auxiliaryDrivSimDisplay", 1},                          // 辅助驾驶模拟显示
    {"fcta", 1},                                             // FCTA(前向目标横穿警告)
    {"BSDWarningLight", 1},                                  // BSD警示灯
    {"reserved_16", 1},                                      // 预留
    {"autohold", 1},                                         // 自动驻车（AUTOHOLD）
    {"parkingBrakeType", 1},                                 // 驻车制动类型
    {"caliperType", 1},                                      // 卡钳类型
    {"backupBrakeControlRCU", 1},                            // RCU备份制动控制器
    {"rvc", 1},                                              // RVC（倒车影像）
    {"avm", 1},                                              // AVM（360环视）
    {"ehb", 1},                                              // EHB(电子液压制动)
    {"reserved_22", 1},                                      // 预留
    {"eps", 1},                                              // 电动助力转向（EPS）
    {"tireSpecification", 4},                                // 轮胎规格
    {"steerMode", 1},                                        // Personalization enable (steering mode)转向模式
    {"turn", 2},                                             // Tuning (calibration parameters)转向助力参数
    {"rearMCUManufacturer", 4},                              // 后电机控制器厂家
    {"frontMCUManufacturer", 4},                             // 前电机控制器厂家
    {"rearMotorManufacturer", 8},                            // (后)电机厂家
    {"gearRatio", 8},                                        // (后)GearRatio(速比)
    {"reserved_17", 8},                                      // 预留
    {"EMSManufacturer", 3},                                  // EMS厂家
    {"EngineManufacturer", 3},                               // 发动机厂家
    {"reserved_18", 2},                                      // 预留
    {"GeneratorManufacturer", 3},                            // 发电机厂家
    {"reserved_19", 5},                                      // 预留
    {"reserved_20", 8}                                       // 预留
};

class CfgVehicleParse {
 public:
    static void DeserVehiclecfgData(std::vector<uint8_t> bits, CfgServerData& dstdata) {
        if (bits.size() != 58) {
            CONFIG_LOG_WARN << "bits:" << bits.size();
            return;
        }

        std::vector<uint8_t> bytes = CfgVehicleParse::DeserBits2Byte(bits);
        for (uint32_t i = 0; i < vehiclecfgkeyvec.size(); i++) {
            std::string key = vehiclecfgkey + vehiclecfgkeyvec[i].first;
            dstdata.cfgParamDataMap_[key].perFlag = false;
            dstdata.cfgParamDataMap_[key].dataType = CFG_DATA_UINT8;
            dstdata.cfgParamDataMap_[key].dataSize = 1;
            dstdata.cfgParamDataMap_[key].lastupdateTime = CfgUtils::GetUTCTime();
            dstdata.cfgParamDataMap_[key].lastupdateClientName = "Diag";
            dstdata.cfgParamDataMap_[key].paramValue = CfgUtils::NumToString(bytes[i]);
        }
        std::string versionkey = vehiclecfgkey + "version";
        dstdata.cfgParamDataMap_[versionkey].perFlag = false;
        dstdata.cfgParamDataMap_[versionkey].dataType = CFG_DATA_STRING;
        dstdata.cfgParamDataMap_[versionkey].dataSize = 1;
        dstdata.cfgParamDataMap_[versionkey].lastupdateTime = CfgUtils::GetUTCTime();
        dstdata.cfgParamDataMap_[versionkey].lastupdateClientName = "Diag";
        dstdata.cfgParamDataMap_[versionkey].paramValue = "4.8";
        CONFIG_LOG_INFO << "bits:" << bits.size() << "  bytes:" << bytes.size();
    }

    static std::vector<uint8_t> SerByte2Bits(const std::vector<uint8_t>& bytes) {
        std::string bitSequence;
        for (uint32_t i = 0; i < vehiclecfgkeyvec.size(); i++) {
            for (int j = vehiclecfgkeyvec[i].second - 1; j >= 0; j--) {
                uint8_t bit = (bytes[i] >> j) & 0x01;
                bitSequence += (bit == 1 ? '1' : '0');
            }
        }
        std::vector<uint8_t> bits;
        for (size_t i = 0; i < bitSequence.length(); i += 8) {
            uint8_t byte = 0;
            for (int j = 0; j < 8; j++) {
                char bitChar = bitSequence[i + j];
                uint8_t bit = static_cast<uint8_t>(bitChar - '0');
                byte = (byte << 1) | bit;
            }
            bits.push_back(byte);
        }
        return bits;
    }
    static std::vector<uint8_t> DeserBits2Byte(const std::vector<uint8_t>& bits) {
        std::string bitSequence;
        for (uint32_t i = 0; i < bits.size(); i++) {
            for (int j = 7; j >= 0; j--) {
                uint8_t bit = (bits[i] >> j) & 0x01;
                bitSequence += (bit == 1 ? '1' : '0');
            }
        }
        std::vector<uint8_t> bytes;
        size_t index = 0;
        for (size_t i = 0; i < vehiclecfgkeyvec.size(); i++) {
            uint8_t byte = 0;
            for (size_t j = 0; j < vehiclecfgkeyvec[i].second; j++) {
                byte = (byte << 1) + (bitSequence[index + j] - '0');
            }
            index += vehiclecfgkeyvec[i].second;
            bytes.push_back(byte);
        }
        return bytes;
    }
};

}  // namespace cfg
}  // namespace netaos
}  // namespace hozon
#endif  // SERVICE_CONFIG_SERVER_INCLUDE_CFG_VEHICLECFG_H_
