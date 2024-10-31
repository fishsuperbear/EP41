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
#include <string.h>
#include <stdbool.h>
#include <stdint.h>

#include "utils.h"
#include "buffer.h"


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

struct buffer {
    size_t read_pos;
    size_t write_pos;
    size_t size;
    size_t capacity;
    memory_t* memory;
};


/* internal static function */

static size_t
buffer_allocate_len(size_t size, size_t capacity)
{
    size_t new_len;

    if (capacity < 64UL) { // the minimum memory is 64 bytes.
        new_len = 64UL;
    }
    else {
        new_len = (capacity > size) ? capacity : size;
    }

    return new_len;
}

static bool
buffer_adjust_len(buffer_t* buffer)
{
    size_t unused_size=0UL;
    size_t used_size=0UL;
    size_t total_size = 0UL;
    bool is_error = true;

    {
        mem_seg_t* node = NULL;
        memory_list_for_each(node, buffer->memory->link, link) {
            total_size += node->size;

            // #1. there is unused data at the end and needs to be resized.
            if (total_size > buffer->size) {
                if (total_size != buffer->capacity) { // #1.1. check if total_size is not equal to capacity.
                    is_error = true;
                }
                else { // #1.2. resize memory segment size.
                    unused_size = total_size - buffer->size;
                    used_size = node->size - unused_size;

                    // #1.2.1. construct a new segment based on the used_size.
                    mem_seg_t* mem_segment = mem_seg_new_form_other(node, 0, used_size);
                    if (mem_segment) {

                        // #1.2.1. insert the new segment before this node.
                         memory_list_insert_before(&node->link, &mem_segment->link);

                         // #1.2.2. unref and remove the node from list.
                         mem_seg_unref(node);
                         memory_list_remove(&node->link);

                         // #1.2.3. end and success.
                         is_error = false;
                         break;
                     }
                }
            } // #2. not needs to be resized.
            else if(total_size == buffer->size) {
                is_error = false;
                break;
            } // #3. do nothing.
            else {
                ;
            }
        }
    }

    return is_error;
}

static int
buffer_segment_write_at_tail(buffer_t* buffer,  size_t offset, const void* data, size_t size, size_t new_len)
{
    MEM_CHECK_POINTER_RET_INTEGER(buffer->memory->link);

    size_t total_size = 0UL;
    size_t remaining_size = 0UL;
    size_t cur_segment_size = 0UL;
    bool is_last_write_end = (((buffer->capacity)-new_len) == (buffer->size)) ? true : false;

    mem_seg_t* node = NULL;
    mem_seg_t* node_prev = NULL;

    // #1. new memory allocated.
    if (new_len > 0) {  // #1.1 write all to the newly allocated memory.
        if (is_last_write_end) {

            // #1.1.1. get the last memory segment.
            node = memory_container_of(buffer->memory->link->prev, node, link);

            // #1.1.2. calculate the position to be written.
            remaining_size = buffer->capacity - offset;
            cur_segment_size = (node->size) -remaining_size;
            if (remaining_size >= size) {
                // #1.1.3. copy data to this segment.
                (void)memcpy(((uint8_t*)(node->data)) + cur_segment_size, data, size);

                // #1.1.4. update write position.
                buffer->write_pos += size;

                // #1.1.5. decide whether to update the size,
                if (buffer->write_pos > buffer->size) {
                    buffer->size += (buffer->write_pos - buffer->size);
                }

                return 0;
            }
        } // #1.2 write to memory in segments.
        else {
            // #1.2.1. get the last two memory segments.
            node_prev = memory_container_of(buffer->memory->link->prev->prev, node_prev, link);
            node = memory_container_of(buffer->memory->link->prev, node, link);

            // #1.2.2. calculate the position to be written.
            remaining_size = buffer->capacity - offset - node->size;
            cur_segment_size = (node_prev->size) -remaining_size;


            // #1.2.3. copy data to previous segment.
            (void)memcpy((uint8_t*)(node_prev->data) + cur_segment_size, data, remaining_size);

            // #1.2.4. copy data to last segment.
            (void)memcpy((uint8_t*)(node->data), (uint8_t*)data + remaining_size, size - remaining_size);

            // #1.2.5. update write position and size.
            buffer->write_pos += size;
            if (buffer->write_pos > buffer->size) {
                buffer->size += (buffer->write_pos - buffer->size);
            }

            return 0;
        }
    } // #2. the original memory is enough.
    else {
        // #2.1. get the last memory segment.
        node = memory_container_of(buffer->memory->link->prev, node, link);

        // #2.2. calculate the position to be written.
        remaining_size = buffer->capacity - offset;
        cur_segment_size = (node->size) -remaining_size;
        if (remaining_size >= size) {
            // #2.3. copy data to last segment.
            (void)memcpy((uint8_t*)(node->data) + cur_segment_size, data, size);

            // #2.4. update write position.
            buffer->write_pos += size;

            // #2.5. decide whether to update the size,
            if (buffer->write_pos > buffer->size) {
                buffer->size += (buffer->write_pos - buffer->size);
            }

            return 0;
        }
    }
    return 1;
}

static int
buffer_segment_write_at(buffer_t* buffer,  size_t offset, const void* data, size_t size)
{
    MEM_CHECK_POINTER_RET_INTEGER(buffer->memory->link);

    size_t total_size;
    size_t remaining_size;
    size_t cur_segment_size;
    size_t data_write_size = 0UL;

    while(size > 0UL) {
        total_size = 0UL;
        remaining_size = 0UL;
        cur_segment_size = 0UL;
        mem_seg_t* node = NULL;

        // #1. find one memory segment according the offset.
        {
            memory_list_for_each(node, buffer->memory->link, link) {
                total_size += node->size;
                if (total_size > offset) {
                    break;
                }
            }
        }

        // #2. write data according size.
        remaining_size = total_size - offset;
        cur_segment_size = (node->size) -remaining_size;
        if (remaining_size >= size) { // #2.1. write to this segment.
            // #2.1.1. copy data to this segment.
            (void)memcpy((uint8_t*)(node->data) + cur_segment_size, (uint8_t*)data + data_write_size, size);

            // #2.1.2. update write position.
            buffer->write_pos += size;

            // #2.1.3. decide whether to update the size,
            // because it is possible to overwrite the original data written.
            // it all depends on the user.
            if (buffer->write_pos > buffer->size) {
                buffer->size += (buffer->write_pos - buffer->size);
            }

            // #2.1.4. no need to do anything. return success.
            return 0;
        } // #2.2 need to write in segments.
        else {
            // #2.2.1 write to the current segment.
            (void)memcpy((uint8_t*)(node->data) + cur_segment_size, (uint8_t*)data + data_write_size, remaining_size);

            // #2.1.2. update write position and size.
            buffer->write_pos += remaining_size;
            if (buffer->write_pos > buffer->size) {
                buffer->size += (buffer->write_pos - buffer->size);
            }

            // #2.2.2 the remaining data is will write to the next segment.
            size   -= remaining_size;
            offset += remaining_size;
            data_write_size += remaining_size;
        }
    }

    return 1;
}

static int
buffer_segment_read_at(buffer_t* buffer,  size_t offset, void* data, size_t size)
{
    MEM_CHECK_POINTER_RET_INTEGER(buffer->memory->link);

    size_t total_size;
    size_t will_read_size;
    size_t will_read_pos;
    size_t data_offset_pos = 0UL;

    while(size > 0UL) {
        total_size = 0UL;
        will_read_size = 0UL;
        will_read_pos = 0UL;
        mem_seg_t* node = NULL;

        // #1. find one memory segment according the offset.
        {
            memory_list_for_each(node, buffer->memory->link, link) {
                total_size += node->size;
                if (total_size > offset) {
                    break;
                }
            }
        }

        // #2. read data according size.
        will_read_size = total_size - offset;
        will_read_pos = (node->size) -will_read_size;
        if (will_read_size >= size) { // #2.1. read to this segment.
            // #2.1.1. copy data from memory segment to user's data.
            (void)memcpy((uint8_t*)data + data_offset_pos, (uint8_t*)(node->data) + will_read_pos, size);

            // #2.1.2. update read position.
            buffer->read_pos += size;

            // #2.1.3. no need to do anything. return success.
            return 0;
        } // #2.2 need to read in segments.
        else {
            // #2.2.1. copy data from memory segment to user's data.
            (void)memcpy((uint8_t*)data + data_offset_pos, (uint8_t*)(node->data) + will_read_pos, will_read_size);

            // #2.1.2. update read position.
            buffer->read_pos += will_read_size;

            // #2.2.2 the remaining data is will read to the next segment.
            size   -= will_read_size;
            offset += will_read_size;
            data_offset_pos += will_read_size;
        }
    }

    return 1;
}


/* external function */

buffer_t*
buffer_new()
{
    buffer_t* buffer = (buffer_t*)calloc(1, sizeof(buffer_t));
    if (!buffer) {
        return NULL;
    }

    buffer->read_pos = 0;
    buffer->write_pos = 0;
    buffer->size = 0;
    buffer->capacity = 0;
    buffer->memory = NULL;
    return buffer;
}

void
buffer_free(buffer_t** buffer)
{
    MEM_CHECK_POINTER_RET_VOID(buffer);
    MEM_CHECK_POINTER_RET_VOID((*buffer));

    free(*buffer);
    (*buffer) = NULL;
}

int
buffer_attach_memory(buffer_t* buffer, memory_t* memory)
{
    MEM_CHECK_POINTER_RET_INTEGER(buffer);
    MEM_CHECK_POINTER_RET_INTEGER(memory);
    MEM_CHECK_POINTER_RET_INTEGER(memory->link);

    if (memory_acquire(memory)) {
        return 1;
    }

    buffer->memory = memory;
    buffer->size = memory_total_size(buffer->memory);
    buffer->capacity = buffer->size;
    memory_ref(buffer->memory);

    return 0;
}

int
buffer_detach_memory(buffer_t* buffer)
{
    MEM_CHECK_POINTER_RET_INTEGER(buffer);
    MEM_CHECK_POINTER_RET_INTEGER(buffer->memory);

    int ret;
    ret = buffer_adjust_len(buffer);

    memory_unref(buffer->memory);
    memory_release(buffer->memory);
    buffer->memory = NULL;
    return ret;
}

int
buffer_write(buffer_t* buffer, const void* data, size_t size)
{
    MEM_CHECK_POINTER_RET_INTEGER(buffer);
    MEM_CHECK_POINTER_RET_INTEGER(buffer->memory);
    MEM_CHECK_POINTER_RET_INTEGER(data);
    MEM_CHECK_SIZE_LENGTH_RET_INTEGER(size);

    return buffer_write_at(buffer, buffer->write_pos, data, size);
}

int
buffer_write_at(buffer_t* buffer,  size_t offset, const void* data, size_t size)
{
    MEM_CHECK_POINTER_RET_INTEGER(buffer);
    MEM_CHECK_POINTER_RET_INTEGER(buffer->memory);
    MEM_CHECK_POINTER_RET_INTEGER(data);
    MEM_CHECK_SIZE_LENGTH_RET_INTEGER(size);

    size_t new_len = 0UL;
    size_t original_capacity = buffer->capacity;
    size_t allocate_capacity = original_capacity;

    // #1. determine whether you need to allocate space.
    while ((offset + size) > allocate_capacity) {
        new_len += buffer_allocate_len(size, allocate_capacity);
        allocate_capacity = original_capacity + new_len;
    }

    // #2. update capacity.
    buffer->capacity += new_len;

    // #3. need allocate memory segment and insert to the list.
    if (new_len > 0U) {
        mem_seg_t* new_segment = mem_seg_new_form_alloc(new_len);
        if (new_segment) {
            memory_append_segment(buffer->memory, new_segment);
            mem_seg_unref(new_segment);
        }
        else {
            return 1;
        }
    }

    // #4. adjust the write pos according offset.
    buffer->write_pos = offset;

    // #5. write data to memory segment.
    if (offset == buffer->size) {
        // #5.1 for performance considerations, there is no need to traversing the list, write to the end directly.
        return buffer_segment_write_at_tail(buffer, offset, data, size, new_len);
    }
    else {
        // #5.2. write data according size and offset
        return buffer_segment_write_at(buffer, offset, data, size);
    }

}


size_t
buffer_get_write_pos(buffer_t* buffer)
{
    MEM_CHECK_POINTER_RET_INTEGER(buffer);
    return buffer->write_pos;
}

void
buffer_set_write_pos(buffer_t* buffer, size_t pos)
{
    MEM_CHECK_POINTER_RET_VOID(buffer);
    buffer->write_pos = pos;
}


int
buffer_read(buffer_t* buffer, void* data, size_t size)
{
    MEM_CHECK_POINTER_RET_INTEGER(buffer);
    MEM_CHECK_POINTER_RET_INTEGER(buffer->memory);
    MEM_CHECK_POINTER_RET_INTEGER(data);
    MEM_CHECK_SIZE_LENGTH_RET_INTEGER(size);

    return buffer_read_at(buffer, buffer->read_pos, data, size);
}

int
buffer_read_at(buffer_t* buffer,  size_t offset, void* data, size_t size)
{
    MEM_CHECK_POINTER_RET_INTEGER(buffer);
    MEM_CHECK_POINTER_RET_INTEGER(buffer->memory);
    MEM_CHECK_POINTER_RET_INTEGER(data);
    MEM_CHECK_SIZE_LENGTH_RET_INTEGER(size);

    if (0 == size) {
        return 1;
    }

    if ((offset + size) > buffer->size) {
        return 1;
    }

    buffer->read_pos = offset;
    return buffer_segment_read_at(buffer, offset, data, size);
}

size_t
buffer_get_read_pos(buffer_t* buffer)
{
    MEM_CHECK_POINTER_RET_INTEGER(buffer);
    return buffer->read_pos;
}

void
buffer_set_read_pos(buffer_t* buffer, size_t pos)
{
    MEM_CHECK_POINTER_RET_VOID(buffer);
    buffer->read_pos = pos;
}

size_t buffer_get_memory_count(buffer_t* buffer)
{
    MEM_CHECK_POINTER_RET_INTEGER(buffer);
    MEM_CHECK_POINTER_RET_INTEGER(buffer->memory);
    MEM_CHECK_POINTER_RET_INTEGER(buffer->memory->link);
    return memory_list_length(buffer->memory->link);
}


/* not implemented */
int
buffer_write_begin_map(buffer_t* buffer, size_t size, bufeer_map_t* buf_map)
{
    return 1;
}

int
buffer_write_begin_map_at(buffer_t* buffer, size_t offset, size_t size, bufeer_map_t* buf_map)
{
    return 1;
}

void
buffer_write_end_map(memory_t* memory, bufeer_map_t* buf_map)
{
}

int
buffer_read_begin_map(buffer_t* buffer, size_t size, bufeer_map_t* buf_map)
{
    return 1;
}

int
buffer_read_begin_map_at(buffer_t* buffer, size_t offset, size_t size, bufeer_map_t* buf_map)
{
    return 1;
}

void
buffer_read_end_map(memory_t* memory, bufeer_map_t* buf_map)
{
}

/*EOF*/

