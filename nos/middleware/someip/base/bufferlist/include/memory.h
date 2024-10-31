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
#ifndef __AP_MEMORY_H__
#define __AP_MEMORY_H__

#include <stddef.h>
#include <sys/uio.h>
#include "list.h"

#ifndef  __cplusplus
#include <stdatomic.h>
#else
#include <atomic>
#define _Atomic(X) std::atomic<X>
#endif


#ifdef  __cplusplus
extern "C" {
#endif


typedef void (*free_func_t)(void*);

typedef struct iovec iovec_t;
typedef struct mem_seg mem_seg_t;
typedef struct memory memory_t;

typedef struct mem_map {
    size_t  iov_cnt;
    iovec_t iov_raw[0];
} mem_map_t;


/* memory segment operation */
mem_seg_t* mem_seg_ref(mem_seg_t* segment);
void mem_seg_unref(mem_seg_t* segment);

mem_seg_t* mem_seg_new_form_alloc(size_t size);
mem_seg_t* mem_seg_new_form_external(void* data, size_t size, free_func_t free_func, void* data_ctx);
mem_seg_t* mem_seg_new_form_other(mem_seg_t* segment, size_t offset, size_t size);

void* mem_seg_data(mem_seg_t* segment);
size_t mem_seg_size(mem_seg_t* segment);


/* memory operation */
memory_t* memory_alloc();

memory_t* memory_ref(memory_t* memory);
void memory_unref(memory_t* memory);

int memory_acquire(memory_t* memory);
void memory_release(memory_t* memory);

mem_map_t* memory_map(memory_t* memory);
void memory_unmap(mem_map_t* mem_map, memory_t* memory);

int memory_append_segment(memory_t* memory, mem_seg_t* segment);
int memory_insert_segment(memory_t* memory, size_t offset, mem_seg_t* segment);
int memory_remove_segment(memory_t* memory, size_t offset, size_t size);
size_t memory_total_size(memory_t* memory);


/* not implemented */
int memory_join_from_segment(memory_t* memory, size_t offset, mem_seg_t* segment);
int memory_join_from_other(memory_t* memory, size_t offset, memory_t* other);

mem_seg_t* memory_extract_to_segment(memory_t* memory, size_t offset, size_t size);
memory_t* memory_extract_to_other(memory_t* memory, size_t offset, size_t size);

mem_seg_t* memory_cut_to_segment(memory_t* memory, size_t offset, size_t size);
memory_t* memory_cut_to_other(memory_t* memory, size_t offset, size_t size);
void memory_remove(memory_t* memory, size_t offset, size_t size);


#ifdef  __cplusplus
} // extern "C"
#endif // __cplusplus
#endif // __AP_MEMORY_H__
/* EOF */
