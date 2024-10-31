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
#ifndef __AP_BUFFER_H__
#define __AP_BUFFER_H__

#include <stddef.h>
#include <unistd.h>
#include "memory.h"


#ifdef  __cplusplus
extern "C" {
#endif

typedef struct buffer buffer_t;

typedef struct buf_map {
    size_t size;
    void* data;
} bufeer_map_t;


buffer_t* buffer_new();
void buffer_free(buffer_t** buf);

int buffer_attach_memory(buffer_t* buffer, memory_t* memory);
int buffer_detach_memory(buffer_t* buffer);

/* write operation */
int buffer_write(buffer_t* buffer, const void* data, size_t size);
int buffer_write_at(buffer_t* buffer,  size_t offset, const void* data, size_t size);
size_t buffer_get_write_pos(buffer_t* buffer);
void buffer_set_write_pos(buffer_t* buffer, size_t pos);


/* read operation */
int buffer_read(buffer_t* buffer, void* data, size_t size);
int buffer_read_at(buffer_t* buffer,  size_t offset, void* data, size_t size);
size_t buffer_get_read_pos(buffer_t* buffer);
void buffer_set_read_pos(buffer_t* buffer, size_t pos);

size_t buffer_get_memory_count(buffer_t* buffer);



/* not implemented */
int buffer_write_begin_map(buffer_t* buffer, size_t size, bufeer_map_t* buf_map);
int buffer_write_begin_map_at(buffer_t* buffer, size_t offset, size_t size, bufeer_map_t* buf_map);
void buffer_write_end_map(memory_t* memory, bufeer_map_t* buf_map);

int buffer_read_begin_map(buffer_t* buffer, size_t size, bufeer_map_t* buf_map);
int buffer_read_begin_map_at(buffer_t* buffer, size_t offset, size_t size, bufeer_map_t* buf_map);
void buffer_read_end_map(memory_t* memory, bufeer_map_t* buf_map);



#ifdef  __cplusplus
} // extern "C"
#endif // __cplusplus
#endif // __AP_BUFFER_H__
/* EOF */
