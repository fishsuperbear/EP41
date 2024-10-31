/*er
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2021. All rights reserved.
 * Description: entity abstract class
 */

#ifndef CANSTACK_E2E_H
#define CANSTACK_E2E_H

#include <string>
#include <thread>
#include <vector>
#include <map>

#include "linux/can.h"
#include "yaml-cpp/yaml.h"
#include "e2e/e2e/include/e2e_sm.h"
#include "e2e/e2exf/include/e2exf.h"
#include "e2e/e2exf_cpp/include/e2exf_impl.h"
#include "log/include/logging.h"

namespace hozon {
namespace netaos {
namespace canstack {

using namespace hozon::netaos::e2e;

struct FaultConfig {
  bool reported = false;
  uint32_t fault_id = 0;
};

class E2ESupConfigType {
 public:
  E2ESupConfigType(const E2EXf_Index& index, const E2EXf_Config& config, uint32_t fault_id) 
    : index_(index), config_(config), counter_(0), start_flag_(false) 
  {
    fault_config_.fault_id = fault_id;
  };

  ~E2ESupConfigType(){};

  const E2EXf_Index& GetIndex(){
    return index_;
  };

  const E2EXf_Config& GetConfig(){
    return config_;
  };

  const std::uint8_t GetCounter(){
    return counter_;
  };

  void IncreaseCounter() {
    if(counter_ < 0x0Fu) {
      counter_++;
    }
    else {
      counter_ = 0u;
    }
  };

  FaultConfig& GetFault() {
    return fault_config_;
  }

  bool GetStartStatus() {
    return start_flag_;
  }

  void RefreshStartStatus() {
    start_flag_ = true;
  }

 private:
  // E2EXf Index
  E2EXf_Index index_;

  // E2EXf Config
  E2EXf_Config config_;

  // as request by hozon standard, self maintained.
  std::uint8_t counter_;

  // fault config
  FaultConfig fault_config_;

  // tag system just startup
  bool start_flag_;
};

class E2ESupervision {
 public:
  static E2ESupervision* Instance();
  E2ESupervision();
  ~E2ESupervision();

  void Init(std::string config_path);

  void Init(YAML::Node& yamlConfig);

  std::uint8_t Protect(can_frame& frame);

  std::uint8_t Protect(canfd_frame& frame);

  std::uint8_t Check(canfd_frame& frame, int32_t& readBytes);

  void SelfTest();

  void SelfTest2();

 private:
  
  std::uint8_t CheckCan(can_frame& frame);

  std::uint8_t CheckCanfd(canfd_frame& frame);

  ProtectResult E2EXf_Protect_DIY(const E2EXf_Index &Index, Payload &Buffer, const std::uint32_t &InputBufferLength,
                            const std::uint32_t &Counter);

  std::uint8_t crc8(std::uint8_t* data_byte, std::uint8_t cb_DATA_BYTE_SIZE);

  bool is_init_{false};
  bool is_enabled_{false};
  std::map<std::uint32_t, std::shared_ptr<E2ESupConfigType>> map_{};
  // void* dl_handle_ = nullptr;
  //TODO: fm agent
  std::shared_ptr<hozon::netaos::log::Logger> log_{hozon::netaos::log::CreateLogger(
    "E2E", "E2E Supervision", static_cast<hozon::netaos::log::LogLevel>(2))};

  static E2ESupervision * instance_;
};

}  // namespace canstack 
}
}  // namespace hozon
#endif  // CANSTACK_E2E_H