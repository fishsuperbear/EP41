#include "logblock_helper/include/log_block_producer.h"

#include <chrono>
#include <ctime>
#include <cstring>
#include <algorithm>

#include "logblock_helper/include/log_block_common_defs.h"

namespace hozon {
namespace netaos {
namespace logblock {

thread_local LogBlockProducer LogBlockProducer::instance_;

LogBlockProducer::LogBlockProducer() {
}

LogBlockProducer::~LogBlockProducer() {
}

LogBlockProducer& LogBlockProducer::Instance() {
    return instance_;
}

bool LogBlockProducer::Init(int data_type, const std::string &appid) {
    auto ret = OpenLogBlockHandle(appid.c_str(), &log_block_handle_);
    if (ret == 0) {
        ret = GetLogBlockWriterInfo(log_block_handle_, &log_block_writer_info_);
        if (ret != 0) {
            return false;
        }
    } else {
        return false;
    }
    inited_ = true;

    return inited_;
}

bool LogBlockProducer::IsInited() const {
    return inited_;
}

LogBlockProducer::WriteStatusCode LogBlockProducer::Write(const std::string &appid,
            const std::string &str, unsigned int data_type,
            unsigned short version) noexcept {
    if (logblock_unlikely(!IsInited())) {
        if (!Init(data_type, appid)) {
            return WriteStatusCode::INIT_FAILED;
        }
    }
    return Write(appid, str.c_str(), str.size(), data_type, version);
}

LogBlockProducer::WriteStatusCode LogBlockProducer::Write(const std::string &appid,
            const char *str, int size, unsigned int data_type,
            unsigned short version) noexcept {
    if (logblock_unlikely(!IsInited())) {
        if (!Init(data_type, appid)) {
            return WriteStatusCode::INIT_FAILED;
        }
    }

    if (size <= 0) {
        return WriteStatusCode::INVALID_DATA;
    }

    auto ret = WriteStatusCode::SUCCESS;
    int body_len = sizeof(LogBody) + size;
    int header_len = sizeof(LogHeader); 

    u32 pwoffset = *(log_block_writer_info_.pwoffset);
    //u32 proffset = *(log_block_writer_info_.proffset);
    u32 block_size = log_block_writer_info_.blocksize;

    // align page
    u32 offset_len = (pwoffset & (block_size - 1));
    u32 jump_offset = 0;
    bool diff = ((header_len + body_len + pwoffset) & (~(ALIGN_PAGE_SIZE - 1))) != (pwoffset & (~(ALIGN_PAGE_SIZE - 1)));
    if (diff) {
         char* occupy_addr = (char*)(log_block_writer_info_.vaddr) + offset_len;
         *occupy_addr = '\a';
         u32 align_offset = ((pwoffset + (ALIGN_PAGE_SIZE - 1)) & ~(ALIGN_PAGE_SIZE - 1));
         jump_offset += align_offset - pwoffset;
         if (jump_offset > 1) {
             memset((char*)(log_block_writer_info_.vaddr) + offset_len + 1, 0, jump_offset - 1);
         }
         offset_len = (align_offset & (block_size - 1));
         ret = WriteStatusCode::RESET_WRITE_POSITION;
    }

    char* addr = (char*)(log_block_writer_info_.vaddr) + offset_len; 
    logblock::LogHeader *header = (logblock::LogHeader*)addr;

    header->magic_num = LOG_HEADER_MAGIC_NUM;
    header->id = ++CURRENT_ID;
    header->version = version;
    header->data_type = data_type;
    header->header_len = header_len;
    header->len = header_len + body_len;
    header->process_id = CURRENT_PROCESS_ID;
    header->thread_id = CURRENT_THREAD_ID;
    header->curr_realtime_ns = GetRealTimeNS();
    header->curr_virtualtime_ns = GetVirtualTimeNS();
    int minlen = std::min(31, (int)appid.size());
    memcpy(header->appid, appid.c_str(), minlen);
    header->appid[minlen] = '\0';

    logblock::LogBody *body = (logblock::LogBody*)(addr + header_len);
    body->reserved = 0;
    memcpy(body->content, str, size);
    body->content[size] = '\0';

    *(log_block_writer_info_.pwoffset) += jump_offset + header->len;

    TagLogBlockDirty(log_block_handle_);

    return ret;
}

} // namespace logblock 
} // namespace netaos
} // namespace hozon
