/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
 *
 * @File: basic_task.h
 * @Date: 2023/07/13
 * @Author: cheng
 * @Desc: --
 */

#ifndef MIDDLEWARE_TOOLS_DATA_COLLECT_COLLECTION_COLLECTION_H_
#define MIDDLEWARE_TOOLS_DATA_COLLECT_COLLECTION_COLLECTION_H_

#include <functional>
#include <string>

#include "utils/include/dc_logger.hpp"
#include "trans_struct.h"
#include "task.h"
#include "yaml-cpp/yaml.h"

namespace hozon {
namespace netaos {
namespace dc {
enum TaskStatus : int;
typedef std::function<void(void)> Callback;

struct CollectFlag {
  bool canCollect = true;
  bool ethCollect = true;
  bool calibrationCollect = true;
  bool faultCollect = true;
  bool logCollect = true;
  bool scCollect = true;
};

class BasicTask {
 public:
  BasicTask(){}
  virtual ~BasicTask() {
    if (cb_ != nullptr) {
      cb_();
    }
  }
  virtual void onCondition(std::string type, char *data, Callback callback) {}
  virtual void configure(std::string type,YAML::Node &node) {}
  virtual void configure(std::string type,DataTrans &node) {}
  virtual void active() {}
  virtual void deactive() {}
  virtual void terminate() {deactive();}
  virtual TaskStatus getStatus() = 0;
  virtual bool getTaskResult(const std::string &type, struct DataTrans&dataStruct) = 0;
  virtual void pause() {}
  virtual void doWhenDestroy(const Callback &callback) { cb_ = callback; }
  static CollectFlag collectFlag;
 private:
  Callback cb_ = nullptr;
};

// CollectFlag BasicTask::collectFlag;

}  // namespace dc
}  // namespace netaos
}  // namespace hozon

#endif  // MIDDLEWARE_TOOLS_DATA_COLLECT_COLLECTION_COLLECTION_H_
