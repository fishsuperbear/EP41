/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description: doip data queue， used to cache diagnostic data
 */
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include "diag/doip/include/base/doip_data_queue.h"

namespace hozon {
namespace netaos {
namespace diag {


DoipDataQueue::DoipDataQueue() : head_(nullptr), tail_(nullptr) {
}

DoipDataQueue::~DoipDataQueue() {
    doip_data_queue_t *node = head_;
    while (head_ != nullptr) {
        node = head_->next;
        delete[] head_->cache_data->data;
        delete head_->cache_data;
        delete head_;
        head_ = node;
    }
}

bool
DoipDataQueue::DoipQueueEmpty() {
    return head_ == nullptr ? true : false;
}

uint32_t
DoipDataQueue::DoipQueueSize() {
    uint32_t queue_size = 0;
    doip_data_queue_t* node = head_;
    while (node != nullptr) {
        queue_size++;
        node = node->next;
    }

    return queue_size;
}

int8_t
DoipDataQueue::DoipInsertQueue(doip_cache_data_t* cache_data) {
    doip_data_queue_t *queue_node = new doip_data_queue_t;
    memset(queue_node, 0, sizeof(doip_data_queue_t));
    queue_node->cache_data = cache_data;
    queue_node->next = nullptr;

    if (nullptr == head_) {
        head_ = tail_ = queue_node;
    } else {
        tail_->next = queue_node;
        tail_ = queue_node;
    }

    return 0;
}

doip_cache_data_t*
DoipDataQueue::DoipPopFrontQueue() {
    if (nullptr == head_) {
        return nullptr;
    }

    doip_data_queue_t* head = head_;
    doip_cache_data_t* data = head_->cache_data;
    head_ = head_->next;

    delete head;
    return data;
}

void
DoipDataQueue::DoipClearQueue() {
    doip_data_queue_t *clear_queue = head_;
    while (clear_queue != nullptr) {
        head_ = clear_queue->next;
        delete[] clear_queue->cache_data->data;
        delete clear_queue->cache_data;
        delete clear_queue;
        clear_queue = head_;
    }
}

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
/* EOF */
