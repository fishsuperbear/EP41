/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description: doip data queue， used to cache diagnostic data
 */

#ifndef MIDDLEWARE_DIAG_DOIP_INCLUDE_BASE_DOIP_DATA_QUEUE_H_
#define MIDDLEWARE_DIAG_DOIP_INCLUDE_BASE_DOIP_DATA_QUEUE_H_

#include <stdint.h>

namespace hozon {
namespace netaos {
namespace diag {

typedef struct doip_cache_data {
    int32_t fd;
    uint32_t data_size;
    char* data;
} doip_cache_data_t;


typedef struct doip_data_queue_ {
    doip_cache_data_t *cache_data;
    struct doip_data_queue_ *next;
} doip_data_queue_t;


class DoipDataQueue {
 public:
    DoipDataQueue();
    ~DoipDataQueue();
    bool DoipQueueEmpty();
    uint32_t DoipQueueSize();
    int8_t DoipInsertQueue(doip_cache_data_t* cache_data);
    doip_cache_data_t* DoipPopFrontQueue();
    void DoipClearQueue();

 private:
    doip_data_queue_t* head_;
    doip_data_queue_t* tail_;
};

}  // namespace diag
}  // namespace netaos
}  // namespace hozon

#endif  // MIDDLEWARE_DIAG_DOIP_INCLUDE_BASE_DOIP_DATA_QUEUE_H_
/* EOF */
