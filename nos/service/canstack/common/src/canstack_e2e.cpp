/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2021. All rights reserved.
 * Description: entity abstract class
 */

#include <iostream>
#include <mutex>
#include <dlfcn.h>
#include "yaml-cpp/yaml.h"
// #include "hz_fm_agent.h"
#include "canstack_e2e.h"
// #include "fault_report.h"
#include "e2e/e2exf_cpp/include/e2exf_config.h"

namespace hozon {
namespace netaos {
namespace canstack {

using namespace hozon::netaos::e2e;

// 声明 E2E_P22ProtectStateType
typedef struct {
    uint8_t Counter;
} E2E_P22ProtectStateType;

// 声明 E2E_P22ConfigType
typedef struct {
    uint16_t DataLength;
    uint8_t DataIDList[16];
    uint8_t MaxDeltaCounter;
    uint16_t Offset;
    bool EnableCRC;
    bool EnableCRCHW;
} E2E_P22ConfigType;

// 声明函数指针 E2E_P22Protect
uint8_t (*E2E_P22Protect)(const E2E_P22ConfigType*, E2E_P22ProtectStateType*, uint8_t*, uint16_t);

std::mutex g_e2e_mutex;

E2ESupervision* E2ESupervision::instance_ = nullptr;

E2ESupervision * E2ESupervision::Instance()
{
    if (nullptr == instance_)
    {
        std::lock_guard<std::mutex> lck(g_e2e_mutex);
        if (nullptr == instance_)
        {
            instance_ = new E2ESupervision();
        }
    }
    return instance_;
}

E2ESupervision::E2ESupervision() {
    // dlopen to load libE2ELib.so and get protect function pointer.
    // dl_handle_ = dlopen("/opt/platform/mdc_platform/lib/libE2ELib.so", RTLD_LAZY|RTLD_GLOBAL);
    // if (!dl_handle_) {
    //   // TODO: fault report.
    //   log_->LogError() << "Cannot dlopen /opt/platform/mdc_platform/lib/libE2ELib.so. dlerror: " << dlerror();
    //   return;
    // }
    
    // E2E_P22Protect = (uint8_t (*)(const E2E_P22ConfigType*, E2E_P22ProtectStateType*, uint8_t*, uint16_t)) dlsym(dl_handle_, "E2E_P22Protect");
    // if (!E2E_P22Protect) {
    //   log_->LogError() << "Cannot get(dlsym) E2E_P22Protect function pointer. error: " << dlerror();
    // }
}

E2ESupervision::~E2ESupervision() {
    // dlclose 卸载动态库
    // if (dl_handle_) {
    //   dlclose(dl_handle_);
    // }
}

void E2ESupervision::Init(std::string config_path) {
    YAML::Node config = YAML::LoadFile(config_path);
    if(config) {
        Init(config);
    } 
}
#define PARSE_YAML_CFG(config, type, key, value) {  \
        if(!config[key]) {                          \
            log_->LogInfo() << "Has no "<< key << " field in config."; \
            break;                                  \
        }                                           \
        else {                                      \
            value = config[key].as<type>();         \
        }                                           \
    }

void E2ESupervision::Init(YAML::Node& yamlConfig)
{
  do
  {
    if(is_init_ == true) {
      break;
    }

    // check if e2e is enabled.
    PARSE_YAML_CFG(yamlConfig, bool, "e2eEnabled", is_enabled_);
    
    if(is_enabled_ == false) {
      log_->LogInfo() << "E2ESupervision Init: e2e disabled.";
      break;
    }

    // iterator through settings for each CAN Message.
    YAML::const_iterator iter;
    for(iter = yamlConfig["canMsgsSettings"].begin(); iter != yamlConfig["canMsgsSettings"].end(); iter++) {
      
      const YAML::Node& Setting = *iter;

      // 1, Get CAN MsgID
      std::uint32_t canMsgId = Setting["canMsgId"].as<std::uint32_t>();
      
      // 2, Get E2E Settings
      E2EXf_ConfigType E2EXfConfig;
      E2EXfConfig.disableEndToEndCheck = Setting["proSettings"]["disableE2ECheck"].as<bool>();
      E2EXfConfig.ProfileConfig.Profile22.MaxDeltaCounter = static_cast<std::uint8_t>(Setting["proSettings"]["<<"]["maxDeltaCounter"].as<std::uint32_t>());
      E2EXfConfig.ProfileConfig.Profile22.Offset          = static_cast<std::uint16_t>(Setting["proSettings"]["<<"]["offset"].as<std::uint32_t>());
      E2EXfConfig.ProfileConfig.Profile22.DataLength = static_cast<std::uint16_t>(Setting["proSettings"]["dataLength"].as<std::uint32_t>());
      E2EXfConfig.Profile = E2EXf_Profile::PROFILE22_CUSTOM;
      E2EXfConfig.disableEndToEndStatemachine = FALSE;
      E2EXfConfig.InPlace = TRUE;
      E2EXfConfig.DataTransformationStatusForwarding = noTransformerStatusForwarding;
      std::array<std::uint32_t, 16> dataIdList = Setting["proSettings"]["dataIdList"].as<std::array<std::uint32_t, 16>>();
      for (size_t i = 0u; i < 16u; i++) {
        // dataIDlist ^ Oxff , for yujie e2e framework follow autosar , 
        // but sensor e2e a little diff,  so dataIdList ^ Oxff reach  e2e check pass.
        E2EXfConfig.ProfileConfig.Profile22.DataIDList[i] = static_cast<std::uint8_t>(dataIdList[i])^0xFF; 
      }
      E2EXfConfig.headerLength = 0 * 8;
      E2EXfConfig.upperHeaderBitsToShift = 0 * 8;
    

      // 3, Get SM Settings
      E2E_SMConfigType E2EXfSMConfig;
      E2EXfSMConfig.WindowSizeInit       = static_cast<std::uint8_t>(Setting["smSettings"]["<<"]["windowSizeInit"].as<std::uint32_t>());
      E2EXfSMConfig.WindowSizeInvalid    = static_cast<std::uint8_t>(Setting["smSettings"]["<<"]["windowSizeInvalid"].as<std::uint32_t>());
      E2EXfSMConfig.WindowSizeValid      = static_cast<std::uint8_t>(Setting["smSettings"]["<<"]["windowSizeValid"].as<std::uint32_t>());
      E2EXfSMConfig.MaxErrorStateInit    = static_cast<std::uint8_t>(Setting["smSettings"]["<<"]["maxErrorStateInit"].as<std::uint32_t>());
      E2EXfSMConfig.MinOkStateInit       = static_cast<std::uint8_t>(Setting["smSettings"]["<<"]["minOkStateInit"].as<std::uint32_t>());
      E2EXfSMConfig.MaxErrorStateInvalid = static_cast<std::uint8_t>(Setting["smSettings"]["<<"]["maxErrorStateInvalid"].as<std::uint32_t>());
      E2EXfSMConfig.MinOkStateInvalid    = static_cast<std::uint8_t>(Setting["smSettings"]["<<"]["minOkStateInvalid"].as<std::uint32_t>());
      E2EXfSMConfig.MaxErrorStateValid   = static_cast<std::uint8_t>(Setting["smSettings"]["<<"]["maxErrorStateValid"].as<std::uint32_t>());
      E2EXfSMConfig.MinOkStateValid      = static_cast<std::uint8_t>(Setting["smSettings"]["<<"]["minOkStateValid"].as<std::uint32_t>());
      E2EXfSMConfig.ClearToInvalid       = Setting["smSettings"]["<<"]["clearToInvalid"].as<bool>();
      E2EXfSMConfig.transitToInvalidExtended = true;

      uint32_t fault_id = 0;
      std::ostringstream ss;
      if (Setting["faultSettings"].IsDefined()) {
        ss << "ID: " << std::hex << canMsgId << " -- start. ";
        uint16_t id = Setting["faultSettings"]["<<"]["id"].as<uint16_t>();
        uint16_t obj = Setting["faultSettings"]["<<"]["obj"].as<uint16_t>();
        fault_id = static_cast<uint32_t>(id << 16u) | obj;
        ss << "ID: " << std::hex << canMsgId << " -- end. ";
      }

      // 4, Add Index and Setting.
      E2EXf_Index index(E2EXfConfig.ProfileConfig.Profile22.DataIDList, canMsgId);

      E2EXf_Config config(E2EXfConfig, E2EXfSMConfig);

      map_.insert({canMsgId, std::make_shared<E2ESupConfigType>(index, config, fault_id)});

      bool add_ret = AddE2EXfConfig(index, config);
      if(add_ret == false) {
        ss << " E2ESupervision Init error, AddE2EXfConfig failed, CanMsgId: " << std::hex << canMsgId;
        log_->LogError() << ss.str();
        break;
      }
    
      ss << " AddE2EXfConfig Successful for CanMsgId: " << std::hex << canMsgId;
      log_->LogInfo() << ss.str();
    }

    if(iter == yamlConfig["canMsgsSettings"].end()) {
      is_init_ = true;
      log_->LogInfo() << "E2ESupervision Init Successful!";
    }

  } while (0);

  return;
}

std::uint8_t E2ESupervision::Protect(can_frame& frame)
{
  std::uint8_t result = 0xFFu;

  do
  {
    // 0, check if e2e enabled.
    if(is_enabled_ == false) {
      result = 0x0u;
      // log_->LogDebug() << "Protect CAN: e2e disabled! can_id: " << frame.can_id;
      break;
    }

    // 1, check if e2e is initialized.
    if(is_init_ == false){
      result = 0x0u;
      // log_->LogError() << "Protect CAN: e2e uninitialized! can_id: " << frame.can_id;
      //TODO: add fm report.
      break;
    }

    // 2, check if e2e needed according to canMsgId.
    auto it = map_.find(frame.can_id);
    if(it == map_.end()) {
      result = 0x0u;
      //log_->LogDebug() << "Protect CAN: canMsgId not found, canMsgId:" << frame.can_id;
      break;
    }

    // 2.1 check if this canMsgId is enabled e2e.
    if(map_[frame.can_id]->GetConfig().GetE2EXfConfig().disableEndToEndCheck == true) {
      result = 0x0u;
      //log_->LogDebug() << "Protect CAN: canmsg e2e disabled, canMsgId: " << frame.can_id;
      break;
    }

    // 3, re-arrange the can data.
    std::uint32_t headerSize = 0u;
    Payload payLoad(headerSize);
    //TODO: add config of crc position.
    std::vector<uint8_t> data1(frame.data, frame.data + frame.can_dlc); //remove crc.
    payLoad.insert(payLoad.end(), data1.begin(), data1.end());

    // 4, write counter.
    std::uint8_t counter = map_[frame.can_id]->GetCounter();

    // payLoad[headerSize - 1] = counter;
    
    payLoad[frame.can_dlc] &= 0xF0u;
    payLoad[frame.can_dlc] |= counter;

    // 5, calc the crc.
    // ProtectResult protectResult = E2EXf_Protect_DIY(map_[frame.can_id]->GetIndex(), payLoad, frame.can_dlc - 1u, counter);
    // // ProtectResult protectResult = E2EXf_Protect(map_[frame.can_id]->GetIndex(), payLoad, frame.can_dlc - 1u, counter);
    // if(protectResult != ProtectResult::OK) {
    //   log_->LogError() << "Protect CAN failed, calc crc error!";
    //   break;
    // }

    // E2E_P22ProtectStateType state {counter};
    // const E2EXf_ConfigType& xfconfig_ = map_[frame.can_id]->GetConfig().GetE2EXfConfig();
    // E2E_P22ConfigType e2e_p22config_ {
    //   xfconfig_.ProfileConfig.Profile22.DataLength,
    //   {0},
    //   xfconfig_.ProfileConfig.Profile22.MaxDeltaCounter,
    //   xfconfig_.ProfileConfig.Profile22.Offset,
    //   // xfconfig_.ProfileConfig.Profile22.EnableCRC,
    //   // xfconfig_.ProfileConfig.Profile22.EnableCRCHW
    //   };
    // memcpy(e2e_p22config_.DataIDList, xfconfig_.ProfileConfig.Profile22.DataIDList.data(), xfconfig_.ProfileConfig.Profile22.DataIDList.size());
    ProtectResult protect_ret = E2EXf_Protect(map_[frame.can_id]->GetIndex(), payLoad, payLoad.size());
    if (protect_ret != ProtectResult::E_OK) {
      log_->LogError() << "Protect CAN failed, calc crc error!";
      break;
    }
    else {
      // log_->LogInfo() << "Protect CAN success. can id: " << frame.can_id;
    }

    // 6, write counter and crc into the can frame.
    frame.data[frame.can_dlc - 2] &= 0xF0u; 
    frame.data[frame.can_dlc - 2] |= (payLoad[frame.can_dlc - 2] & 0x0Fu); // counter
    frame.data[frame.can_dlc - 1]  = payLoad[frame.can_dlc - 1]; // crc

    // 7, increase the counter.
    map_[frame.can_id]->IncreaseCounter();

    result = 0x0u;

  } while (0);

  return result;
}

std::uint8_t E2ESupervision::Protect(canfd_frame& frame)
{
  std::uint8_t result = 0xFFu;

  do
  {
    // 0, check if e2e enabled.
    if(is_enabled_ == false) {
      result = 0x0u;
      //log_->LogDebug() << "Protect CANFD: e2e disabled!";
      break;
    }

    // 1, check if e2e is initialized.
    if(is_init_ == false) {
      result = 0x0u;
      //log_->LogError() << "Protect CANFD: e2e uninitialized!";
      //TODO: add fm report.
      break;
    }

    // 2, check if e2e needed according to canMsgId.
    auto it = map_.find(frame.can_id);
    if(it == map_.end()) {
      result = 0x0u;
      // log_->LogDebug() << "Protect CANFD: canMsgId not found, canMsgId: " << frame.can_id;
      break;
    }

    // 2.1 check if this canMsgId is enabled e2e.
    if(map_[frame.can_id]->GetConfig().GetE2EXfConfig().disableEndToEndCheck == true) {
      result = 0x0u;
      // log_->LogDebug() << "Protect CANFD: canmsg e2e disabled, canMsgId: " << frame.can_id;
      break;
    }

    // 3, re-arrange the can data.
    std::uint32_t headerSize = 0u;
    Payload payLoad(headerSize);
    //TODO: add config of crc position.
    // if(frame.len > 8u) {
    //   std::vector<uint8_t> data1(frame.data, frame.data + 7); //remove crc.
    //   std::vector<uint8_t> data2(frame.data + 8, frame.data + frame.len);

    //   payLoad.insert(payLoad.end(), data1.begin(), data1.end());      
    //   payLoad.insert(payLoad.end(), data2.begin(), data2.end());
    // }
    // else {
      std::vector<uint8_t> data3(frame.data, frame.data + frame.len); //remove crc.

      payLoad.insert(payLoad.end(), data3.begin(), data3.end());
    // }

    // 4, write counter
    std::uint8_t counter = map_[frame.can_id]->GetCounter();

    // payLoad[headerSize - 1] = counter;
    
    if(frame.len > 8u) {
      payLoad[8] &= 0xF0u;
      payLoad[8] |= (counter & 0x0Fu);
    }
    else {
      payLoad[frame.len] &= 0xF0u;
      payLoad[frame.len] |= (counter & 0x0Fu);
    }

    // 5, calc the crc.
    // ProtectResult protect_ret = E2EXf_Protect_DIY(map_[frame.can_id]->GetIndex(), payLoad, frame.len - 1u, counter);
    // // ProtectResult protect_ret = E2EXf_Protect(map_[frame.can_id]->GetIndex(), payLoad, frame.len - 1u, counter);
    // if(protect_ret != ProtectResult::OK) {
    //   log_->LogError() << "Protect CANFD failed, calc crc error!";
    //   break;
    // }

    // E2E_P22ProtectStateType state {counter};
    // const E2EXf_ConfigType& xfconfig_ = map_[frame.can_id]->GetConfig().GetE2EXfConfig();
    // E2E_P22ConfigType e2e_p22config_ {
    //   xfconfig_.ProfileConfig.Profile22.DataLength,
    //   {0},
    //   xfconfig_.ProfileConfig.Profile22.MaxDeltaCounter,
    //   xfconfig_.ProfileConfig.Profile22.Offset
    //   // xfconfig_.ProfileConfig.Profile22.EnableCRC,
    //   // xfconfig_.ProfileConfig.Profile22.EnableCRCHW
    //   };
    // memcpy(e2e_p22config_.DataIDList, xfconfig_.ProfileConfig.Profile22.DataIDList.data(), xfconfig_.ProfileConfig.Profile22.DataIDList.size());
    
    ProtectResult protect_ret = E2EXf_Protect(map_[frame.can_id]->GetIndex(), payLoad, payLoad.size());

    if (protect_ret != ProtectResult::E_OK) {
      log_->LogError() << "Protect CANFD failed, calc crc error!";
      break;
    }
    else {
      // log_->LogInfo() << "Protect CANFD successfully. can id: " << frame.can_id;
    }

    //log_->LogDebug() << "E2EXf_Protect CanMsgID: " << frame.can_id << " crc: " << static_cast<int>(payLoad[0]) << " counter: " << static_cast<int>(payLoad[1]);

    // 6, write counter and crc into the can frame.
    if(frame.len > 8u) {
      frame.data[6] &= 0xF0u;
      frame.data[6] |= (payLoad[6] & 0x0Fu);
      frame.data[7] = payLoad[7];
    }
    else {
      frame.data[frame.len - 2] &= 0xF0u;
      frame.data[frame.len - 2] |= (payLoad[frame.len - 2] & 0x0Fu);
      frame.data[frame.len - 1]  = payLoad[frame.len - 1];
    }
   
    // 7, increase the counter.
    map_[frame.can_id]->IncreaseCounter();

    result = 0x0u;

  } while (0);

  return result;
}

std::uint8_t E2ESupervision::CheckCan(can_frame& frame)
{
  std::uint8_t result = 0xFFu;

  do
  {
    // 0, check if e2e enabled.
    if(is_enabled_ == false) {
      result = 0x0u;
      // log_->LogInfo() << "Check CANFD: e2e disabled!";
      break;
    }

    // 1, check if e2e is initialized.
    if(is_init_ == false){
      result = 0x0u;
      // log_->LogInfo() << "Check CANFD:, e2e uninitialized!";
      //TODO: add fm report.
      break;
    }

    // 2, check if e2e needed according to canMsgId.
    auto it = map_.find(frame.can_id);
    if(it == map_.end()) {
      result = 0x0u;
      // log_->LogInfo() << "Check CANFD: canMsgId not found, canMsgId: " << frame.can_id;
      break;
    }

    // 2.1 check if this canMsgId is enabled e2e.
    if(map_[frame.can_id]->GetConfig().GetE2EXfConfig().disableEndToEndCheck == true) {
      result = 0x0u;
      // log_->LogInfo() << "Check CANFD: canmsg e2e disabled, canMsgId: " << frame.can_id;
      break;
    }

    // 3, re-arrange the can data.
    std::uint32_t headerSize = 0u;
    Payload payLoad(headerSize);
    //TODO: add config of crc position.
    std::vector<uint8_t> data(frame.data, frame.data + frame.can_dlc); //
    payLoad.insert(payLoad.end(), data.begin(), data.end());
    
    // 4, writer e2e header
    // payLoad[0] = frame.data[frame.can_dlc - 1];
    // payLoad[1] = frame.data[frame.can_dlc - 2] & 0x0F;
    // payLoad[1] |= 0xF0u; // for huawei bug.

    // 5, check the crc.
    hozon::netaos::e2e::CheckResult checkResult = 
          hozon::netaos::e2e::E2EXf_Check(map_[frame.can_id]->GetIndex(), payLoad, payLoad.size());

    if(((checkResult.GetProfileCheckStatus() != E2EXf_PCheckStatusType::E2E_P_OK)
         || (checkResult.GetSMState() != E2EXf_SMStateType::E2E_SM_VALID))
      && (map_[frame.can_id]->GetStartStatus() == true)) {
      log_->LogError() << "Check CAN failed, ProfileCheckStatus: " << static_cast<int>(checkResult.GetProfileCheckStatus()) << 
                         ", SMState: " << static_cast<int>(checkResult.GetSMState()) << 
                         ", crc expected: " << static_cast<int>(payLoad[0]) << 
                         ", counter expected: " << static_cast<int>(payLoad[1] & 0xFu) << 
                         ", crc received: " << static_cast<int>(frame.data[7]) <<
                         ", counter received: " << static_cast<int>(frame.data[6] & 0xFu) <<
                         ", canMsgId: " << frame.can_id;

      // Report fault. Only in case of:
      // 1. sm state is invalid.
      // 2. wrong sequence (more than max_delta mesage is lost)
      if (checkResult.GetSMState() == E2EXf_SMStateType::E2E_SM_INVALID) {
        auto& fault_config = map_[frame.can_id]->GetFault();
        if (!fault_config.reported) {
          // hozon::canstack::CanBusReport::Instance().ReportSensorFaultAsync((fault_config.fault_id & 0xFFFF0000u) >> 16u, 
          //                                             (fault_config.fault_id & 0x0000FFFFu), 1, hozon::fm::USE_FM_CHANNEL); 
          // fault_config.reported = true;
          // log_->LogError() << "Reprot fault: " << ((fault_config.fault_id & 0xFFFF0000u) >> 16u) << " " <<  (fault_config.fault_id & 0x0000FFFFu) << ", canMsgId: " << frame.can_id;
        }
      }
      // else if ((checkResult.GetSMState() == E2EXf_SMStateType::E2E_SM_INVALID) && (checkResult.GetProfileCheckStatus() == ProfileCheckStatus::kWrongSequence)) {
      //   auto& fault_config = map_[frame.can_id]->GetFault();
      //   if (!fault_config.reported) {
      //     hozon::canstack::CanBusReport::Instance().ReportSensorFaultAsync((fault_config.fault_id & 0xFFFF0000u) >> 16u, 
      //                                                 (fault_config.fault_id & 0x0000FFFFu), 1, hozon::fm::USE_FM_CHANNEL); 
      //     fault_config.reported = true;
      //     log_->LogError() << "Reprot fault: " << ((fault_config.fault_id & 0xFFFF0000u) >> 16u) << " " <<  (fault_config.fault_id & 0x0000FFFFu) << ", canMsgId: " << frame.can_id;
      //   }
      // }
      else {

      }
      break;
    }
    else {
      // Clear e2e fault report status.
      map_[frame.can_id]->GetFault().reported = false;
      // log_->LogInfo() << "Check CAN successful, canMsgId: " << frame.can_id << " counter: " << static_cast<int>(payLoad[1]);
      // check if this frame is first group frame
      if(map_[frame.can_id]->GetStartStatus() == false) {
        map_[frame.can_id]->RefreshStartStatus();
        break;
      }
    }

    result = 0u;
    
  } while (0);

  return result;
}

std::uint8_t E2ESupervision::CheckCanfd(canfd_frame& frame)
{
  std::uint8_t result = 0xFFu;

  do
  {
    // 0, check if e2e enabled.
    if(is_enabled_ == false) {
      result = 0x0u;
      // log_->LogInfo() << "Check CANFD: e2e disabled!";
      break;
    }

    // 1, check if e2e is initialized.
    if(is_init_ == false){
      result = 0x0u;
      // log_->LogInfo() << "Check CANFD:, e2e uninitialized!";
      //TODO: add fm report.
      break;
    }

    // std::stringstream ss;
                            
    // ss << "can id:0x" << std::hex << frame.can_id << " data: " ;
    // for(int index = 0; index < frame.len; index++) {
    //     ss << " " << std::hex << frame.data[index];
    // }
    // log_->LogError() << ss.str();
    // printf(" can id:%x  data:", frame.can_id);
    // for(int i = 0; i < frame.len ; i++ ) {
    //     printf(" %2x", frame.data[i]);
    // }
    // printf(" \r\n"); 
    // 2, check if e2e needed according to canMsgId.
    auto it = map_.find(frame.can_id);
    if(it == map_.end()) {
      result = 0x0u;
      // log_->LogInfo() << "Check CANFD: canMsgId not found, canMsgId: " << std::hex << frame.can_id;
      break;
    }

    // 2.1 check if this canMsgId is enabled e2e.
    if(map_[frame.can_id]->GetConfig().GetE2EXfConfig().disableEndToEndCheck == true) {
      result = 0x0u;
      // log_->LogInfo() << "Check CANFD: canmsg e2e disabled, canMsgId: " << std::hex << frame.can_id;
      break;
    }

    // 3, re-arrange the can data.
    Payload payLoad(frame.data, frame.data + frame.len);
    // std::uint32_t headerSize = 0u;
    // Payload payLoad(headerSize);
    //TODO: add config of crc position.
    // if(frame.len > 8u) {
    //   std::vector<uint8_t> data1(frame.data, frame.data + 7); //remove crc.
    //   std::vector<uint8_t> data2(frame.data + 8, frame.data + frame.len);

    //   payLoad.insert(payLoad.end(), data1.begin(), data1.end());
    //   payLoad.insert(payLoad.end(), data2.begin(), data2.end());
    // }
    // else {
    //   std::vector<uint8_t> data3(frame.data, frame.data + frame.len - 1); //remove crc.

    //   payLoad.insert(payLoad.end(), data3.begin(), data3.end());
    // }     

    // // 4, writer e2e header
    // if(frame.len > 8u) {
    //   payLoad[0] = frame.data[7];
    //   payLoad[1] = frame.data[6] & 0x0F;
    //   // // payLoad[1] |= 0xF0u; // for huawei bug.
    // }
    // else {
    //   payLoad[0] = frame.data[frame.len - 1];
    //   payLoad[1] = frame.data[frame.len - 2] & 0x0F;
    //   // // payLoad[1] |= 0xF0u; // for huawei bug.
    // }

    // 5, check the crc.
    hozon::netaos::e2e::CheckResult checkResult = 
        hozon::netaos::e2e::E2EXf_Check(map_[frame.can_id]->GetIndex(), payLoad, payLoad.size());

    if(((checkResult.GetProfileCheckStatus() != E2EXf_PCheckStatusType::E2E_P_OK) 
          || (checkResult.GetSMState() != E2EXf_SMStateType::E2E_SM_VALID))
      &&  (map_[frame.can_id]->GetStartStatus() == true)) {
      std::ostringstream oss;
      oss << "Check CANFD failed, ProfileCheckStatus: " << static_cast<int>(checkResult.GetProfileCheckStatus()) << 
                         ", SMState: " << static_cast<int>(checkResult.GetSMState()) << 
                         ", crc expected: " << static_cast<int>(payLoad[7]) << 
                         ", counter expected: " << static_cast<int>(payLoad[6] & 0xFu) << 
                         ", crc received: " << static_cast<int>(frame.data[7]) <<
                         ", counter received: " << static_cast<int>(frame.data[6] & 0xFu) <<
                         ", canMsgId: " << std::hex << frame.can_id;
      log_->LogError() << oss.str();
      // Report fault. Only in case of:
      // 1. sm state is invalid.
      // 2. wrong sequence (more than max_delta mesage is lost)
      if (checkResult.GetSMState() == E2EXf_SMStateType::E2E_SM_INVALID) {
        auto& fault_config = map_[frame.can_id]->GetFault();
        if (!fault_config.reported) {
            // hozon::canstack::CanBusReport::Instance().ReportSensorFaultAsync((fault_config.fault_id & 0xFFFF0000u) >> 16u, 
            //             (fault_config.fault_id & 0x0000FFFFu), 1, hozon::fm::USE_FM_CHANNEL); 
          fault_config.reported = true;
          log_->LogError() << "Reprot fault: " << ((fault_config.fault_id & 0xFFFF0000u) >> 16u) << " " <<  (fault_config.fault_id & 0x0000FFFFu) << ", canMsgId: " << frame.can_id;
        }
      }
      // else if ((checkResult.GetSMState() == E2EXf_SMStateType::E2E_SM_INVALID) && (checkResult.GetProfileCheckStatus() == ProfileCheckStatus::kWrongSequence)) {
      //   auto& fault_config = map_[frame.can_id]->GetFault();
      //   if (!fault_config.reported) {
      //     // hozon::fm::HzFMAgent::Instance()->ReportFaultAsync(
      //     //   hozon::fm::HzFMAgent::Instance()->GenFault((fault_config.fault_id & 0xFFFF0000u) >> 16u,
      //     //                                             (fault_config.fault_id & 0x0000FFFFu), hozon::fm::USE_FM_CHANNEL), 1);
      //     hozon::canstack::CanBusReport::Instance().ReportSensorFaultAsync((fault_config.fault_id & 0xFFFF0000u) >> 16u, 
      //                                                 (fault_config.fault_id & 0x0000FFFFu), 1, hozon::fm::USE_FM_CHANNEL); 
      //     fault_config.reported = true;
      //     log_->LogError() << "Reprot fault: " << ((fault_config.fault_id & 0xFFFF0000u) >> 16u) << " " <<  (fault_config.fault_id & 0x0000FFFFu) << ", canMsgId: " << frame.can_id;
      //   }
      // }
      else {

      }
      break;
    }
    else {
      // Clear e2e fault report status.
      map_[frame.can_id]->GetFault().reported = false;
      // log_->LogInfo() << "Check CANFD successful, canMsgId: " << frame.can_id << ", counter: " << static_cast<int>(payLoad[1]) << ", canMsgId: " << frame.can_id;
      
      // check if this frame is first group frame 
      if(map_[frame.can_id]->GetStartStatus() == false) {
        map_[frame.can_id]->RefreshStartStatus();
        break;
      }
    }
    result = 0u;

  } while (0);

  return result;
}

std::uint8_t E2ESupervision::Check(canfd_frame& frame, int32_t& readBytes)
{
  std::uint8_t result = 0x0u;

  if (readBytes == sizeof(can_frame)) {
      result = CheckCan(*(reinterpret_cast<can_frame*>(&frame)));
  }
  else {
      result = CheckCanfd(frame);
  }

  return result;
}

void E2ESupervision::SelfTest()
{
  log_->LogInfo() << "+++++++++++++++++ start e2e selftest 1 ++++++++++++++++";

  //prepare data 
  Payload data = {0x00, 0xf4, 0x57, 0x12, 0x2d, 0x23, 0x00,  
                  0x96, 0x3f, 0xe6, 0x01, 0x9f, 0xf3, 0xfe, 0x00, 
                  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
                  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
  const std::size_t dataLength = data.size();  // without e2e header.

  //header of each profile can be obtained in table 1. 
  const std::uint32_t headerSize = 2U;

  Payload testDataP22(headerSize); 
  testDataP22.insert(testDataP22.end(), data.begin(), data.end()); 

  E2EXf_Index index = map_[0x440u]->GetIndex();
  
  for(std::uint16_t i = 0u; i < 17u; i++)
  {
    ProtectResult protectResult = E2EXf_Protect(index, testDataP22, dataLength);

    map_[0x440u]->IncreaseCounter();

    hozon::netaos::e2e::CheckResult CheckResult =E2EXf_Check(index, testDataP22, dataLength + headerSize);

    // if(testDataP22[2+6] < 0x0Fu){
    //   testDataP22[2+6]++;
    // }
    // else {
    //   testDataP22[2+6] = 0u;
    // }

    std::cout << std::hex << "protectResult: " << static_cast<int>(protectResult) << 
                            " crc: " << static_cast<int>(testDataP22[0]) << 
                            " counter: " << static_cast<int>(testDataP22[1]) << 
                            " checkResult: " << static_cast<int>(CheckResult.GetProfileCheckStatus()) << 
                            " checkResult sm: " << static_cast<int>(CheckResult.GetSMState()) << std::endl;
  }

  return;
}

void E2ESupervision::SelfTest2()
{
  log_->LogInfo() << "+++++++++++++++++ start e2e selftest 2 ++++++++++++++++";

  // prepare CANFD frame
  std::array<std::uint8_t, 32> data{0x00, 0xf4, 0x57, 0x12, 0x2d, 0x23, 0x00, 0x00,
                                    0x96, 0x3f, 0xe6, 0x01, 0x9f, 0xf3, 0xfe, 0x00, 
                                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
                                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
  canfd_frame frame;
  frame.can_id = 0x440u;
  frame.len = 32u;
  memcpy(frame.data, data.data(), frame.len);

  for(std::uint16_t i = 0u; i < 17u; i++)
  {
    // protect
    std::uint8_t ret = Protect(frame);
    std::cout << std::hex << "SelfTest2 Protect ret: " << static_cast<int>(ret) 
                          << " crc: " << static_cast<int>(frame.data[7])
                          << " counter: " << static_cast<int>(frame.data[6])
                          << std::endl;

    // check
    if(ret !=0x0u)
    {
      std::cout << "SelfTest2 Protect failed." << std::endl;
      // break;
    }

    // ret = CheckCanfd(frame);
    // std::cout << std::hex << "SelfTest2 Check ret: " << static_cast<int>(ret) << std::endl;

  }

  return;
}

ProtectResult E2ESupervision::E2EXf_Protect_DIY(const E2EXf_Index &Index, Payload &Buffer, const std::uint32_t &InputBufferLength,
                            const std::uint32_t &Counter) {
  
  ProtectResult ret = ProtectResult::HardRuntimeError;

  do
  {
    if(Buffer.size() != (InputBufferLength + 2)) {
      log_->LogError() << "E2EXf_Protect_DIY: length error.";
      break;
    }

    // Buffer[1] = Counter | 0xF0u;
    Buffer[1] = Counter;

    Buffer.push_back(Index.GetDataIDList()[Counter]);

    std::uint8_t checksum = crc8(Buffer.data() + 1, Buffer.size() - 1);

    // std::cout << "checksum: " << std::hex << static_cast<int>(checksum) << std::dec << std::endl;

    Buffer[0] = checksum;

    ret = ProtectResult::E_OK;

  } while (0);
  
  return ret;
}

std::uint8_t E2ESupervision::crc8(std::uint8_t* data_byte, std::uint8_t cb_DATA_BYTE_SIZE)
{   
  uint8_t CB_CRC_POLY = 0x2Fu;
  uint8_t crc = 0xFFu;  // initial value = 0xFF
  for( uint8_t byteindex_=0; byteindex_ < cb_DATA_BYTE_SIZE; ++byteindex_ ) {
      crc ^= *(data_byte + byteindex_);
      for(uint8_t bitindex_=0; bitindex_ < 8u; ++bitindex_ ){
          if( (crc & 0x80u) != 0 )
              crc = (crc << 1u) ^ CB_CRC_POLY;
          else
              crc = (crc << 1u);
      }
  }
  return ~crc;    //return inverse value
}


}  // namespace canstack
}
}  // namespace hozon

