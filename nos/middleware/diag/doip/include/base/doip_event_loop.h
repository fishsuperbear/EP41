/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description: doip event loop
 */

#ifndef MIDDLEWARE_DIAG_DOIP_INCLUDE_BASE_DOIP_EVENT_LOOP_H_
#define MIDDLEWARE_DIAG_DOIP_INCLUDE_BASE_DOIP_EVENT_LOOP_H_

#include <stdint.h>

#include "diag/doip/include/base/doip_select.h"

namespace hozon {
namespace netaos {
namespace diag {

enum DOIP_EVENT_INFO {
    DOIP_EVENT_READABLE = 0x01,
    DOIP_EVENT_WRITABLE = 0x02,
    DOIP_EVENT_HANGUP   = 0x04,
    DOIP_EVENT_ERROR    = 0x08
};


using eventloop_callback = std::function<int32_t(int32_t, uint32_t, void*)>;

typedef struct doip_event_source {
    int32_t fd;
    void *data;
    eventloop_callback callbak_func;
} doip_event_source_t;


class DoipEventLoop {
 public:
    DoipEventLoop();
    ~DoipEventLoop();
    doip_event_source_t* SourceCreate(int32_t fd, eventloop_callback func, void *data);
    void SourceDestroy(doip_event_source_t* source);
    int32_t SourceAdd(doip_event_source_t* source, int32_t mask);
    int32_t SourceRemove(doip_event_source_t* source, bool close_flag);
    int32_t SourceUpdate(doip_event_source_t* source, int32_t mask);
    int32_t Dispatch(int32_t timeout);
};

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
#endif  // MIDDLEWARE_DIAG_DOIP_INCLUDE_BASE_DOIP_EVENT_LOOP_H_
/* EOF */
