/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description: doip connection
 */

#ifndef MIDDLEWARE_DIAG_DOIP_INCLUDE_BASE_DOIP_CONNECTION_H_
#define MIDDLEWARE_DIAG_DOIP_INCLUDE_BASE_DOIP_CONNECTION_H_

#include <sys/socket.h>
#include <stdint.h>
#include <netinet/in.h>

namespace hozon {
namespace netaos {
namespace diag {

#define DOIP_MAX_BUFFER_SIZE     1500

typedef struct doip_buffer {
    char data[DOIP_MAX_BUFFER_SIZE];
    ssize_t data_size;
} doip_buffer_t;



class DoipConnection {
 public:
    explicit DoipConnection(int32_t fd);
    ~DoipConnection();
    DoipConnection(const DoipConnection&) = delete;
    DoipConnection &operator=(const DoipConnection&) = delete;
    int32_t GetFd();
    void SetFd(int32_t fd);
 private:
    int32_t fd_;

 public:
    doip_buffer_t in_;
    doip_buffer_t out_;
};

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
#endif  // MIDDLEWARE_DIAG_DOIP_INCLUDE_BASE_DOIP_CONNECTION_H_
/* EOF */
