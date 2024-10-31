/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description: doip select
 */

#ifndef MIDDLEWARE_DIAG_DOIP_INCLUDE_BASE_DOIP_SELECT_H_
#define MIDDLEWARE_DIAG_DOIP_INCLUDE_BASE_DOIP_SELECT_H_

#include <sys/select.h>
#include <stdint.h>
#include <mutex>
#include <list>

#include "diag/doip/include/base/doip_thread.h"

namespace hozon {
namespace netaos {
namespace diag {

#define DOIP_EV_READ     0x01
#define DOIP_EV_WRITE    0x02

#define DOIP_EV_NORML    0x00
#define DOIP_EV_INTER    0x01

#define DOIP_SEL_ADD     0x00
#define DOIP_SEL_DEL     0x01
#define DOIP_SEL_MOD     0x02

typedef union event_data {
    void *ptr;
    int32_t fd;
    uint32_t u32;
    uint64_t u64;
} event_data_t;

typedef struct doip_event {
    int32_t fd;
    int32_t ev_type;
    int32_t events;
    event_data_t data;
} doip_event_t;


class DoipSelect {
 public:
    static DoipSelect *Instance();
    void Create();
    void Destroy();
    int32_t Control(int32_t type, int32_t fd, doip_event_t *event);
    int32_t Dispatch(doip_event_t *event, int32_t length, int32_t timeout);
    int32_t Notify();

 private:
    DoipSelect();
    ~DoipSelect();
    DoipSelect(const DoipSelect &);
    DoipSelect & operator = (const DoipSelect &);

    int32_t InternalSocket(int32_t notify_fd[2]);
    int32_t InitNotify();
    int32_t SelectAdd(int32_t fd, doip_event_t *event);
    int32_t SelectDel(int32_t fd);
    doip_event_t *SelectFind(int32_t fd);
    int32_t SelectResponse();

    static DoipSelect *instancePtr_;
    static std::mutex instance_mtx_;

    int32_t max_fd_;
    size_t fd_setsize_;
    int32_t notify_fd_[2];
    fd_set *event_readset_in_;
    fd_set *event_writeset_in_;
    fd_set *event_readset_out_;
    fd_set *event_writeset_out_;
    std::list<doip_event_t*> event_list_;
    std::recursive_mutex mtx_;
};

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
#endif  // MIDDLEWARE_DIAG_DOIP_INCLUDE_BASE_DOIP_SELECT_H_
/* EOF */
