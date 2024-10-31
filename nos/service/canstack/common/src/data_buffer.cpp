/*
* Copyright (c) Hozon Auto Co., Ltd. 2021-2021. All rights reserved.
* Description: socket can interface data buffer
*/

#include <mutex>
#include <memory>
#include <cstring>

#include "data_buffer.h"
// #include "can_stack_utils.h"

namespace hozon {
namespace netaos {
namespace canstack {


// const unsigned char CAN_BUFFER_NO_FLUSHED_DATA = 0xFFU;
// const unsigned char CAN_BUFFER_FLUSHED_DATA_A = 0x00U;
// const unsigned char CAN_BUFFER_FLUSHED_DATA_B = 0x01U;

DataBuffer* DataBuffer::sinstance_ = nullptr;
std::mutex g_data_buffer_mutex;

DataBuffer& DataBuffer::Instance() {
    std::lock_guard<std::mutex> lck (g_data_buffer_mutex);

    if (!sinstance_) {
        sinstance_ = new DataBuffer;
    }

    return *sinstance_;
}

DataBuffer::DataBuffer()
: notified_(false)
, buf_(nullptr){

}

DataBuffer::~DataBuffer() {
    std::lock_guard<std::mutex> lck (g_data_buffer_mutex);
    if (buf_) {
        delete[] buf_;
        buf_ = nullptr;
    }
}

bool DataBuffer::WaitData(int indx, int seconds) {


    std::unique_lock<std::mutex> cond_lock(g_data_buffer_mutex);
    auto pred = [this, indx]() -> bool {return (data_map_.find(indx) != data_map_.end());};

    bool ret = false;
    if (seconds) {
        ret = cv_.wait_for(cond_lock, std::chrono::seconds(seconds), pred);
    }
    else {
        cv_.wait(cond_lock, pred);
        ret = true;
    }

    return ret;
}

bool DataBuffer::WaitData(int* indexes, int count, int seconds) {
    std::unique_lock<std::mutex> cond_lock(g_data_buffer_mutex);
    auto pred = [this, indexes, count]() -> bool {
        bool found = false;
        for (int i = 0; i < count; ++i) {
            if (data_map_.find(indexes[i]) != data_map_.end()) {
                found = true;
                break;
            }
        }

        return found;
    };

    bool ret = false;
    if (seconds) {
        ret = cv_.wait_for(cond_lock, std::chrono::seconds(seconds), pred);
    }
    else {
        cv_.wait(cond_lock, pred);
        ret = true;
    }

    return ret;
}

bool DataBuffer::WaitDataNotification(int index, int seconds) {
    std::unique_lock<std::mutex> cond_lock(g_data_buffer_mutex);
    auto pred = [this]() -> bool { return notified_;};

    bool ret = false;
    if (seconds) {
        ret = cv_.wait_for(cond_lock, std::chrono::seconds(seconds), pred);
    }
    else {
        cv_.wait(cond_lock, pred);
        ret = true;
    }
    notified_ = false;

    return ret;
}
bool DataBuffer::WriteData(int indx, unsigned char* data, int bytes, int startbit, bool notify) {
    bool ret = false;
    {
        std::lock_guard<std::mutex> lck (g_data_buffer_mutex);
        // HAF_LOG_WARN << "pos = " << data_map_[indx]->pos ;
        if (data_map_.find(indx) != data_map_.end()) {
            if(data_map_[indx]->pos >= defaultObjsize) {//already in
                // HAF_LOG_WARN << "clear buffer ";
                // data_map_.erase(indx);
                // data_map_[indx] = std::make_shared<BufNode>(defaultObjsize);
                memset(data_map_[indx]->buf, 0, defaultObjsize);
                data_map_[indx]->pos = 0; 
            }
            //data_map_[indx] = std::make_shared<BufNode>(defaultObjsize);
        } else {
            data_map_[indx] = std::make_shared<BufNode>(defaultObjsize);
        }
        //easy overflow
        // HAF_LOG_WARN << "pos = " << data_map_[indx]->pos ;
        if(data_map_[indx]->pos + bytes > defaultObjsize) {
            // HAF_LOG_ERROR << "buf overlow attention! size = " << data_map_[indx]->pos + bytes;
        } else {
            ::memcpy(data_map_[indx]->buf + startbit, data, bytes);
            data_map_[indx]->pos += bytes;
            ret = true;
        }
/*
        else {
            data_map_[indx] = std::make_shared<BufNode>(defaultObjsize);
        }

        if ((data_map_.find(indx) != data_map_.end()) && data_map_[indx]->buf && (data_map_[indx]->size == bytes)) {
            ::memcpy(data_map_[indx]->buf, data, bytes);
            ret = true;
        }
        */
    }

    if (notify) {
        cv_.notify_all();
    }

    return ret;
}
void DataBuffer::NotifyAll() {
    cv_.notify_all();
}

void DataBuffer::ClearAll() {
    //HAF_LOG_INFO << "clear data map";
    data_map_.clear();
}

bool DataBuffer::WriteData(int indx, unsigned char* data, int bytes, bool notify) {
    bool ret = false;
    {
        std::lock_guard<std::mutex> lck (g_data_buffer_mutex);
        if (data_map_.find(indx) != data_map_.end()) {
            if (data_map_[indx]->size != bytes) {
                data_map_.erase(indx);
                data_map_[indx] = std::make_shared<BufNode>(bytes);
            }
        }
        else {
            data_map_[indx] = std::make_shared<BufNode>(bytes);
        }

        if ((data_map_.find(indx) != data_map_.end()) && data_map_[indx]->buf && (data_map_[indx]->size == bytes)) {
            ::memcpy(data_map_[indx]->buf, data, bytes);
            ret = true;
        }
    }

    if (notify) {
        // hozon::canstack::CanStackUtils::RecordTimeMillis(4);
        notified_ = true;
        cv_.notify_all();
    }

    return ret;
}
// uint64_t DataBuffer::LengthMask(uint8_t length)
// {
//     uint64_t lmask = 0x00000000;
//     for(auto i = 0; i < length;i ++){
//         lmask = (lmask << 1) | 0x00000001;
//     }
//     return lmask;
// }
// uint64_t DataBuffer::ReadDataByPosition(int indx, int frame, uint8_t start_bit, uint8_t length)
// {
//     std::lock_guard<std::mutex> lck (g_data_buffer_mutex);
//     uint8_t offset = 0;
//     uint64_t m_data = 0;
//     uint8_t right_mv_bit = 0;
//     for(int i = frame*8;i < (frame+1)*8;i++) {
//         m_data = (m_data << 8) | data_map_[indx]->buf[i];
//     }
//     offset = start_bit/8;
//     right_mv_bit = start_bit%8;
//     right_mv_bit = right_mv_bit + 8*(7-offset);
//     m_data = m_data >> right_mv_bit;
//     m_data = m_data & LengthMask(length);
//     return m_data;
// }
bool DataBuffer::ReadData(int indx, unsigned char* data, int bytes, bool del_flag) {

    std::lock_guard<std::mutex> lck (g_data_buffer_mutex);

    bool ret = false;

    if ((data_map_.find(indx) != data_map_.end()) && data_map_[indx]->buf && (data_map_[indx]->size <= bytes)) {
        ::memcpy(data, data_map_[indx]->buf, data_map_[indx]->size);
        if (del_flag) {
            data_map_.erase(indx);
        }
        ret = true;
    }
    // hozon::canstack::CanStackUtils::RecordTimeMillis(5);

    return ret;
}

bool DataBuffer::HasData(int indx) {
    std::lock_guard<std::mutex> lck (g_data_buffer_mutex);
    bool has_data = (data_map_.find(indx) != data_map_.end());
    return has_data;
}

size_t DataBuffer::GetDataMapSize()
{
    return data_map_.size();
}

} // namespace canstack
}
} // namespace hozon
