/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2021. All rights reserved.
 * Description: socket can interface canbus monitor
 */

#ifndef CANBUS_MONITOR_H
#define CANBUS_MONITOR_H

#include <condition_variable>
#include <mutex>
#include <string>
#include <vector>

#include "can_parser.h"

namespace hozon {
namespace netaos {
namespace canstack {


typedef enum ResCode {
  RES_SUCCESS = 0,
  RES_PASER_PTR_NULL = -1,
  RES_SOCKET_INIT_FAILED = -2,
  RES_SET_SOCKET_OPT_ERR = -3,
  RES_CAN_READ_ERR = -4,
  RES_CAN_WRITE_ERR = -5,
  RES_ERROR = -99
} ResCode_t;

typedef enum FaultID {
  FAULT_CAN_INIT_ERR = 600,
  FAULT_TX_TIMEOUT,
  FAULT_CRTL_ERR,
  FAULT_PROT_ERR,
  FAULT_TRX_ERR,
  FAULT_ACK_ERR,
  FAULT_BUSOFF_ERR,
  FAULT_BUS_ERR,
} FaultID_t;

typedef struct can_info {
  int can_fd; 
  std::string can_port;
  CanParser *can_parser;
} can_info_struct;
class CanbusMonitor {
 public:
  CanbusMonitor();
  ~CanbusMonitor();

  int Init(const std::string &canDevice, CanParser *canParser);
  int Init(const std::vector<std::string> &canDevice, CanParser* canParserPtr);
  int InitSock(const std::string& canDevice, CanParser*& can_parser_ptr);
  int SetRecvTimeout(const std::string &canDevice, const struct timeval &tv) const;
  int SetCanFiLter(const std::string &canDevice, const struct can_filter &filter) const;
  int SetCanFilters(const int can_fd, const std::vector<can_filter> &filters) const;
  int ReadCan(int can_fd, canfd_frame &receiveFrame, struct timeval &tstamp,
              std::int32_t &readBytes) const;
  int GetSocketCanfd(const std::string &canDevice);
  void SetSocketCanNonBlock(int fd);
  int CloseSocketCan(int sock_can);
  void StartCanbusMonitorThread();
  std::string GetCurrCanDevice(int fd);

 private:
  void CanDisposeTreadCallback();
  bool IsErrorCan(uint32_t can_id, std::string canPort);

  int sock_can_;
  can_info_struct can_info_t;
  std::vector<can_info_struct> sock_list_;
  fd_set can_fds_;
  int maxfd_;
  std::string can_name_;
  bool quit_flag_;
  std::mutex mtx_;
  std::condition_variable cv_;
};

}  // namespace canstack
}
}  // namespace hozon
#endif  // CANBUS_MONITOR_H
