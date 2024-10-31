/**

* Copyright @ 2020 - 2027 iAuto Software(Shanghai) Co., Ltd.

* All Rights Reserved.

*

* Redistribution and use in source and binary forms, with or without

* modification, are NOT permitted except as agreed by

* iAuto Software(Shanghai) Co., Ltd.

*

* Unless required by applicable law or agreed to in writing, software

* distributed under the License is distributed on an "AS IS" BASIS,

* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

*/
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <pthread.h>

#include "utils.h"
#include "memory.h"


#define MEM_CHECK_IS_MAP(MEMORY) \
    bool is_map = MEMORY->mem_is_map; \
    if (is_map) { \
        MLOGE("\033[1;40;31mUT[%s][%d]: check map error. \n\033[0m", __FUNCTION__, __LINE__); \
        return 1; \
    }

static pthread_mutex_t  mem_acquire_mutex = PTHREAD_MUTEX_INITIALIZER;

struct mem_seg {
    atomic_int ref_cnt;
    size_t size;
    free_func_t free_func;
    void* data_ctx;
    void* data;
    memory_list_t link;
};

struct memory {
    atomic_int ref_cnt;
    atomic_bool mem_is_acquire;
    bool mem_is_map;
    memory_list_t* link;
};


mem_seg_t*
mem_seg_ref(mem_seg_t* segment)
{
    MEM_CHECK_POINTER_RET_PTR(segment);
    atomic_fetch_add_explicit(&(segment->ref_cnt), 1, memory_order_acq_rel);
    return segment;
}

void
mem_seg_unref(mem_seg_t* segment)
{
    MEM_CHECK_POINTER_RET_VOID(segment);

    atomic_int ref_cnt;
    ref_cnt = atomic_fetch_sub_explicit(&(segment->ref_cnt), 1, memory_order_acq_rel);
    if (1 == ref_cnt) {
        if (segment->free_func) {
            if (segment->data_ctx) {
                MLOGD("[%s]. [size: %lu][ref: %d].\n",__FUNCTION__, segment->size, segment->ref_cnt);
                segment->free_func(segment->data_ctx);
                segment->data_ctx = NULL;
            }

            free(segment);
            segment = NULL;
            return;
        }
    }
}

mem_seg_t*
mem_seg_new_form_alloc(size_t size)
{
    MEM_CHECK_SIZE_LENGTH_RET_PTR(size);

    mem_seg_t* segment = calloc(1, sizeof(mem_seg_t));
    if (!segment) {
        return NULL;
    }

    segment->data = calloc(1, size);
    if (!segment->data) {
        free(segment);
        return NULL;
    }

    segment->size = size;
    segment->free_func = free;
    segment->data_ctx = segment->data;
    atomic_store_explicit(&(segment->ref_cnt), 1, memory_order_acq_rel);

    return segment;
}

mem_seg_t*
mem_seg_new_form_external(void* data, size_t size, free_func_t free_func, void* data_ctx)
{
    MEM_CHECK_POINTER_RET_PTR(data);
    MEM_CHECK_SIZE_LENGTH_RET_PTR(size);

    mem_seg_t* segment = calloc(1, sizeof(mem_seg_t));
    if (!segment) {
        return NULL;
    }

    segment->data = data;
    segment->size = size;
    segment->free_func = free_func;
    segment->data_ctx = data_ctx;
    atomic_store_explicit(&(segment->ref_cnt), 1, memory_order_acq_rel);

    return segment;
}

mem_seg_t*
mem_seg_new_form_other(mem_seg_t* other, size_t offset, size_t size)
{
    MLOGD("[%s]. [offset: %lu][size: %lu].\n",__FUNCTION__, offset, size);
    MEM_CHECK_POINTER_RET_PTR(other);
    MEM_CHECK_SIZE_LENGTH_RET_PTR(size);

    if ((offset + size) > other->size) {
        return NULL;
    }

    mem_seg_t* segment = calloc(1, sizeof(mem_seg_t));
    if (!segment) {
        return NULL;
    }

    mem_seg_ref(other);
    segment->data = (void*)(((uint8_t*)(other->data)) + offset);
    segment->size = size;
    segment->free_func = (free_func_t)(mem_seg_unref);
    segment->data_ctx = other;
    atomic_store_explicit(&(segment->ref_cnt), 1, memory_order_acq_rel);

    return segment;
}

void*
mem_seg_data(mem_seg_t* segment)
{
    MEM_CHECK_POINTER_RET_PTR(segment);
    return segment->data;
}

size_t
mem_seg_size(mem_seg_t* segment)
{
    MEM_CHECK_POINTER_RET_INTEGER(segment);
    return segment->size;
}

memory_t*
memory_alloc()
{
    memory_t* memory = (memory_t*)calloc(1, sizeof(memory_t));
    if (!memory) {
        return NULL;
    }

    memory->link = (memory_list_t*)calloc(1, sizeof(memory_list_t));
    if (!memory->link) {
        free(memory);
        memory = NULL;
        return NULL;
    }


    memory->mem_is_map = false;
    memory->mem_is_acquire = false;
    memory_list_init(memory->link);
    atomic_store_explicit(&(memory->ref_cnt), 1, memory_order_acq_rel);
    return memory;
}

static void
memory_destory(memory_t* memory)
{
    MEM_CHECK_POINTER_RET_VOID(memory->link);
    if (!memory_list_empty(memory->link)) {
        // free memory segment.
        mem_seg_t* node = NULL;
        mem_seg_t* tmp = NULL;
        memory_list_for_each_safe(node, tmp, memory->link, link) {
            memory_list_remove(&node->link);
            mem_seg_unref(node);
        }
    }

    // free memory.
    free(memory->link);
    memory->link = NULL;
    free(memory);
    memory = NULL;
}

memory_t*
memory_ref(memory_t* memory)
{
    MEM_CHECK_POINTER_RET_PTR(memory);
    atomic_fetch_add_explicit(&(memory->ref_cnt), 1, memory_order_acq_rel);
    return memory;
}

void
memory_unref(memory_t* memory)
{
    MEM_CHECK_POINTER_RET_VOID(memory);

    atomic_int ref_cnt;
    ref_cnt = atomic_fetch_sub_explicit(&(memory->ref_cnt), 1, memory_order_acq_rel);
    if (1 == ref_cnt) {
        memory_destory(memory);
    }
}

int
memory_acquire(memory_t* memory)
{
    atomic_bool mem_is_acquire;

    pthread_mutex_lock(&mem_acquire_mutex);
    mem_is_acquire = atomic_load_explicit(&(memory->mem_is_acquire), memory_order_acq_rel);
    if (mem_is_acquire) {
        pthread_mutex_unlock(&mem_acquire_mutex);
        return 1;
    }

    atomic_store_explicit(&(memory->mem_is_acquire), true, memory_order_acq_rel);
    pthread_mutex_unlock(&mem_acquire_mutex);

    return 0;
}

void
memory_release(memory_t* memory)
{
    atomic_store_explicit(&(memory->mem_is_acquire), false, memory_order_acq_rel);
}

mem_map_t*
memory_map(memory_t* memory)
{
    MEM_CHECK_POINTER_RET_PTR(memory);
    MEM_CHECK_POINTER_RET_PTR(memory->link);
    MEM_CHECK_LIST_EMPETY_RET_PTR(memory->link);

    int list_len;
    int iter_cnt;
    mem_seg_t* node;

    list_len = memory_list_length(memory->link);
    mem_map_t* mem_map = (mem_map_t*)calloc(1, sizeof(mem_map_t) + ((list_len)*sizeof(iovec_t)));
    if (!mem_map) {
        return NULL;
    }

    node = NULL;
    iter_cnt = 0;
    mem_map->iov_cnt = list_len;
    memory_list_for_each(node, memory->link, link) {
        mem_map->iov_raw[iter_cnt].iov_len = node->size;
        mem_map->iov_raw[iter_cnt].iov_base = node->data;
        iter_cnt++;
   }

    memory->mem_is_map = true;
    return mem_map;
}

void
memory_unmap(mem_map_t* mem_map, memory_t* memory)
{
    MEM_CHECK_POINTER_RET_VOID(mem_map);
    free(mem_map);
    mem_map = NULL;

    MEM_CHECK_POINTER_RET_VOID(memory);
    memory->mem_is_map = false;
}

int
memory_append_segment(memory_t* memory, mem_seg_t* segment)
{
    MEM_CHECK_POINTER_RET_INTEGER(memory);
    MEM_CHECK_IS_MAP(memory);
    MEM_CHECK_POINTER_RET_INTEGER(memory->link);
    MEM_CHECK_POINTER_RET_INTEGER(segment);

    mem_seg_ref(segment);
    memory_list_insert_tail(memory->link, &segment->link);
    return 0;
}

int
memory_insert_segment(memory_t* memory, size_t offset, mem_seg_t* segment)
{
    MEM_CHECK_POINTER_RET_INTEGER(memory);
    MEM_CHECK_IS_MAP(memory);
    MEM_CHECK_POINTER_RET_INTEGER(memory->link);
    MEM_CHECK_POINTER_RET_INTEGER(segment);

    size_t total_size;
    size_t offset_relative;
    size_t first_segment_size;
    size_t last_segment_size;


    // #1. if the list is empty.
    if (memory_list_empty(memory->link)) {
        if (0 != offset) {
            return 1;
        }
        else { // #1.1 insert the memory segment at the tail/head.
            mem_seg_ref(segment);
            memory_list_insert_tail(memory->link, &segment->link);
            return 0;
        }
    }

    // #2. if the list is not empty.
    total_size = 0;
    {
        mem_seg_t* node = NULL;
        mem_seg_t* temp = NULL;

        memory_list_for_each_safe(node, temp, memory->link, link) {
            total_size += node->size;
            if (total_size > offset) { // #2.1 insert the memory segment in the middle of other memory segment.
                offset_relative = total_size - offset;
                first_segment_size  = node->size - offset_relative;
                last_segment_size = offset_relative;

                // #2.1.1. construct two new segment based on the offset.
                mem_seg_t* mem_segment_first = mem_seg_new_form_other(node, 0, first_segment_size);
                mem_seg_t* mem_segment_last = mem_seg_new_form_other(node, first_segment_size, last_segment_size);

                // #2.1.2. insert mem_segment_first and mem_segment_last before and after node.
                if (mem_segment_first) {
                    memory_list_insert_before(&node->link, &mem_segment_first->link);
                }
                if (mem_segment_last) {
                    memory_list_insert_after(&node->link, &mem_segment_last->link);
                }

                // #2.1.3. unref and remove node.
                mem_seg_unref(node);
                memory_list_remove(&node->link);

                // #2.1.4. insert new node.
                mem_seg_ref(segment);
                memory_list_insert_before(&mem_segment_last->link, &segment->link);
                break;
            }
            else if(total_size == offset) { // #2.2. insert the memory segment after the given element.
                mem_seg_ref(segment);
                memory_list_insert_after(&node->link, &segment->link);
                break;
            }
            else {
                ; // #2.3. TODO: for other.
            }
        }
    }

    // #3. offset exceed total size.
    if (offset > total_size) {
        MLOGE("[%s][%d]. Error: [offset:%lu] > [total_size: %lu]\n",__FUNCTION__, __LINE__, offset, total_size);
        return 1;
    }
    return 0;
}

int
memory_remove_segment(memory_t* memory, size_t offset, size_t size)
{
    MEM_CHECK_POINTER_RET_INTEGER(memory);
    MEM_CHECK_IS_MAP(memory);
    MEM_CHECK_SIZE_LENGTH_RET_INTEGER(size);
    MEM_CHECK_POINTER_RET_INTEGER(memory->link);
    MEM_CHECK_LIST_EMPETY_RET_INTEGER(memory->link);


    size_t list_length;
    size_t total_size;
    size_t offset_relative;
    size_t first_segment_size;
    size_t last_segment_size;
    size_t relative_size;

    // check the length is legal.
    list_length = memory_total_size(memory);
    if (offset + size > list_length) {
        return 1;
    }


    while (1) {
        total_size = 0U;
        {
            mem_seg_t* node = NULL;
            mem_seg_t* temp = NULL;

            memory_list_for_each_safe(node, temp, memory->link, link) {
                total_size += node->size;
                if (total_size > offset) { // #1. find the node corresponding to the offset.
                    offset_relative = total_size - offset;
                    first_segment_size  = node->size - offset_relative;

                    // #1.1 offset in the first node.
                    if (first_segment_size + size < node->size) {
                        last_segment_size = node->size - first_segment_size - size;
                        mem_seg_t* mem_segment_last = mem_seg_new_form_other(node, first_segment_size + size, last_segment_size);
                        if (mem_segment_last) { // #1.1.1 create a segment according this node and insert to this list.
                            memory_list_insert_after(&node->link, &mem_segment_last->link);

                            // #1.1.1.1 unref the node and remove form the list.
                            mem_seg_unref(node);
                            memory_list_remove(&node->link);
                            return 0;
                        }
                        return 1; // #1.1.1.2 failed case.
                    }
                    else if (first_segment_size + size == node->size) {  // #1.2 offset is at the begging of a certain node and offset equal to this node size.

                        // #1.2.1 unref the node and remove form the list.
                        mem_seg_unref(node);
                        memory_list_remove(&node->link);
                        return 0;
                    }
                    else { // #1.3 offset spans several nodes.
                        last_segment_size = node->size - first_segment_size;

                        if (last_segment_size == node->size) { // #1.3.1 the size to be deleted covers the entire node.
                        }
                        else { // #1.3.2 part of the offset is in this node.
                            mem_seg_t* mem_segment_first = mem_seg_new_form_other(node, 0, first_segment_size);
                            if (mem_segment_first) {
                                memory_list_insert_before(&node->link, &mem_segment_first->link);
                            }
                        }

                        // unref the node and remove form the list.
                        mem_seg_unref(node);
                        memory_list_remove(&node->link);

                        // update the size to be deleted.
                        size -= last_segment_size;
                        break;
                    }
                }
                else if (total_size < offset) {
                    ;  // #2. TODO: for other.
                }
                else {
                    ;  // #3. TODO: for other.
                }
            }
        }
    }


    return 1;
}

size_t
memory_total_size(memory_t* memory)
{
    MEM_CHECK_POINTER_RET_INTEGER(memory);
    size_t memory_size = 0;
    {
        mem_seg_t* node = NULL;
        memory_list_for_each(node, memory->link, link) {
            memory_size += node->size;
        }
    }

    return memory_size;
}



/* not implemented */
int
memory_join_from_segment(memory_t* memory, size_t offset, mem_seg_t* segment)
{
    return 1;
}

int
memory_join_from_other(memory_t* memory, size_t offset, memory_t* other)
{
    return 1;
}

mem_seg_t*
memory_extract_to_segment(memory_t* memory, size_t offset, size_t size)
{
    return NULL;
}

memory_t*
memory_extract_to_other(memory_t* memory, size_t offset, size_t size)
{
    return NULL;
}

mem_seg_t*
memory_cut_to_segment(memory_t* memory, size_t offset, size_t size)
{
    return NULL;
}

memory_t*
memory_cut_to_other(memory_t* memory, size_t offset, size_t size)
{
    return NULL;
}

void
memory_remove(memory_t* memory, size_t offset, size_t size)
{
}

