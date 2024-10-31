/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description: doip event loop
 */

#include <errno.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <cstddef>

#include "diag/doip/include/base/doip_event_loop.h"
#include "diag/doip/include/base/doip_logger.h"

namespace hozon {
namespace netaos {
namespace diag {


#define DOIP_MAX_EVENT_SIZE  32
#define ARRAY_LENGTH(a) (sizeof (a) / sizeof (a)[0])


DoipEventLoop::DoipEventLoop() {
}

DoipEventLoop::~DoipEventLoop() {
}


doip_event_source_t*
DoipEventLoop::SourceCreate(int32_t fd, eventloop_callback func, void *data) {
    doip_event_source_t* source = new doip_event_source_t;
    source->fd = fd;
    source->callbak_func = func;
    source->data = data;

    return source;
}

void
DoipEventLoop::SourceDestroy(doip_event_source_t* source) {
    if (nullptr != source) {
        delete source;
    }
}

int32_t
DoipEventLoop::SourceAdd(doip_event_source_t* source, int32_t mask) {
    doip_event_t ep;

    memset(&ep, 0, sizeof ep);
    if (mask & DOIP_EVENT_READABLE) {
        ep.events |= DOIP_EV_READ;
    }
    if (mask & DOIP_EVENT_WRITABLE) {
        ep.events |= DOIP_EV_WRITE;
    }
    ep.data.ptr = source;

    if (DoipSelect::Instance()->Control(DOIP_SEL_ADD, source->fd, &ep) < 0) {
        DOIP_ERROR << "<DoipEventLoop> source add select ctl error!";
        return -1;
    }

    DOIP_INFO << "<DoipEventLoop> source add fd:" << source->fd;

    return 0;
}

int32_t
DoipEventLoop::SourceRemove(doip_event_source_t* source, bool close_flag) {
    DOIP_INFO << "<DoipEventLoop> source remove fd:" << source->fd;
    if (source->fd < 0) {
        return -1;
    }

    int32_t res = DoipSelect::Instance()->Control(DOIP_SEL_DEL, source->fd, NULL);
    if ((0 == res) && close_flag) {
        close(source->fd);
        source->fd = -1;
    }

    return 0;
}

int32_t
DoipEventLoop::SourceUpdate(doip_event_source_t* source, int32_t mask) {
    doip_event_t ep;

    memset(&ep, 0, sizeof ep);
    if (mask & DOIP_EVENT_READABLE) {
        ep.events |= DOIP_EV_READ;
    }
    if (mask & DOIP_EVENT_WRITABLE) {
        ep.events |= DOIP_EV_WRITE;
    }
    ep.data.ptr = source;

    return DoipSelect::Instance()->Control(DOIP_SEL_MOD, source->fd, &ep);
}

int32_t
DoipEventLoop::Dispatch(int32_t timeout) {
    doip_event_t ep[DOIP_MAX_EVENT_SIZE];
    errno = 0;

    int32_t count = DoipSelect::Instance()->Dispatch(ep, ARRAY_LENGTH(ep), timeout);
    if (count < 0) {
        DOIP_ERROR << "<DoipEventLoop> dispatch errno:" << errno << ", message:" << strerror(errno);
        return -1;
    }

    // DOIP_DEBUG << "<DoipEventLoop> dispatch start count: " << count;

    for (int32_t i = 0; i < count; i++) {
        doip_event_source_t *source = reinterpret_cast<doip_event_source_t *>(ep[i].data.ptr);
        if (source == NULL) {
            continue;
        }
        if (source->fd != -1) {
            int32_t mask = 0;
            if (ep[i].events & DOIP_EV_READ) {
                mask |= DOIP_EVENT_READABLE;
            }
            if (ep[i].events & DOIP_EV_WRITE) {
                mask |= DOIP_EVENT_WRITABLE;
            }

            source->callbak_func(source->fd, mask, source->data);
        }
    }

    // DOIP_DEBUG << "<DoipEventLoop> dispatch end.";

    return 0;
}


}  // namespace diag
}  // namespace netaos
}  // namespace hozon
/* EOF */
