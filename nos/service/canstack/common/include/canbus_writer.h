/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2021. All rights reserved.
 * Description: socket can interface canbus writer
 */

#ifndef CANBUS_WRITER_H
#define CANBUS_WRITER_H

#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "can_parser.h"

namespace hozon {
namespace netaos {
namespace canstack {

class CanbusWriter {
 public:
  static CanbusWriter* Instance();
  ~CanbusWriter();

  static int Init(const std::string& canDevice);
  static int WriteCan(int fd, const can_frame& sendFrame);
  static int WriteCan(std::vector<int> fd_list, const can_frame &sendFrame);
  static int WriteCanfd(int fd, canfd_frame& sendFrame);
  static int WriteCanfd(std::vector<int> fd_list, canfd_frame &sendFrame);
  static int SetSocketCanLoop(int fd, int enableFlag);
  static void SetSocketCanNonBlock(int fd);
  static void CloseSocketCan(int fd);

 private:
  CanbusWriter();

  static CanbusWriter* sinstance_;
};

}  // namespace canstack
}
}  // namespace hozon
#endif  // CANBUS_WRITER_H
