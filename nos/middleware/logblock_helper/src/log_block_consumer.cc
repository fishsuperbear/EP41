#include "logblock_helper/include/log_block_consumer.h"

#include <sys/uio.h>

#include <cstdlib>
#include <cstring>
#include <ctime>
#include <sstream>

#include "logblock_helper/include/log_block_common_defs.h"

namespace hozon {
namespace netaos {
namespace logblock {

LogBlockConsumer::LogBlockConsumer() {
    srand(time(NULL));
}

LogBlockConsumer::~LogBlockConsumer() {
    for (auto& thread : threads_) {
        thread.join();
    }
}

LogBlockConsumer& LogBlockConsumer::Instance() {
    static LogBlockConsumer log_block_consumer;
    return log_block_consumer;
}

bool LogBlockConsumer::Start(int thread_num, const CallBackFunc &data_callback_func,
            const ErrorCallBackFunc &error_callback_func) {
    if (inited_) {
        return false;
    }

    thread_num_ = thread_num;
    data_callback_func_ = data_callback_func;
    error_callback_func_ = error_callback_func;

    if (thread_num_ <= 0) {
        return false;
    }
    for (int i = 0; i < thread_num_; ++i) {
        threads_.emplace_back([this]{
            this->Run();
        });
    }
    inited_ = true;

    return true;
}

void LogBlockConsumer::Run() {
    LogBlockReaderThreadHandle log_block_reader_thread_handle;
    if (!OpenLogBlockThreadHandle(log_block_reader_thread_handle)) {
        return;
    }

    LogBlockReaderInfo log_block_reader_info;
    while (running_.load()) {
       if (!UpdateLogBlockReaderInfo(log_block_reader_thread_handle, log_block_reader_info)) {
           continue;
       }

       HandleData(log_block_reader_info);
    }

    FinishLogBlockRead(log_block_reader_thread_handle);
}

void LogBlockConsumer::HandleData(LogBlockReaderInfo &log_block_reader_info) {
    u32 begin = log_block_reader_info.roffset_begin;
    u32 end = log_block_reader_info.roffset_end;

    u32 read_size = log_block_reader_info.roffset_end - log_block_reader_info.roffset_begin;
    if (read_size == 0) {
        return;
    }

    if (read_size > log_block_reader_info.blocksize) {
        read_size = ((log_block_reader_info.blocksize * 3) >> 2);
        begin = end - read_size;
    }

    u32 align_read_begin = (begin & (log_block_reader_info.blocksize - 1));
    u32 align_read_end = (end & (log_block_reader_info.blocksize - 1));

    iovec iov[4];
    if ((begin & (~(log_block_reader_info.blocksize - 1))) !=
                ((begin + read_size) & (~(log_block_reader_info.blocksize - 1)))) {
        ParseData((char*)(log_block_reader_info.vaddr), align_read_begin, log_block_reader_info.blocksize);
        int remain_size = read_size - (log_block_reader_info.blocksize - align_read_begin);
        ParseData((char*)(log_block_reader_info.vaddr), 0, remain_size);
    } else {
        ParseData((char*)(log_block_reader_info.vaddr), align_read_begin, align_read_end);
    }
}

void LogBlockConsumer::ParseData(const char *data_addr, int start_parse_pos, int end_parse_pos) {
    while (start_parse_pos < end_parse_pos) {
        if (*(char*)(data_addr + start_parse_pos) == '\a') {
            int old_start_parse_pos = start_parse_pos;
            start_parse_pos = (start_parse_pos + (ALIGN_PAGE_SIZE - 1)) & ~(ALIGN_PAGE_SIZE - 1);
            if (old_start_parse_pos == start_parse_pos) {
                start_parse_pos += ALIGN_PAGE_SIZE;

                error_callback_func_("invalid data exists. skip this page");
            }
            continue;
        }
        const char *vaddr = data_addr + start_parse_pos;
        LogHeader *header = (LogHeader*)vaddr;
        int header_len = header->header_len;
        if (header->magic_num != LOG_HEADER_MAGIC_NUM) {
            if (error_callback_func_ != nullptr) {
                std::stringstream message;
                message << "magic number don't match,"
                    << header->magic_num
                    << " != "
                    << LOG_HEADER_MAGIC_NUM
                    << ", will handle next page...";
                error_callback_func_(message.str());
            }
            // jump to next page
            if (start_parse_pos % ALIGN_PAGE_SIZE != 0) {
                start_parse_pos = (start_parse_pos + (ALIGN_PAGE_SIZE - 1)) & ~(ALIGN_PAGE_SIZE - 1);
            } else {
                start_parse_pos += ALIGN_PAGE_SIZE;
            }
            continue;
        }
       
        if (header->len > 0) {
            start_parse_pos += header->len;
        } else {
            if (start_parse_pos % ALIGN_PAGE_SIZE != 0) {
                start_parse_pos = (start_parse_pos + (ALIGN_PAGE_SIZE - 1)) & ~(ALIGN_PAGE_SIZE - 1);
            } else {
                start_parse_pos += ALIGN_PAGE_SIZE;
            }
            error_callback_func_("the length of header is unnormal, skip this page");
        }

        LogBody *body = (LogBody*)(vaddr + header_len);
        data_callback_func_(header, body);
    }
}

bool LogBlockConsumer::UpdateLogBlockReaderInfo(LogBlockReaderThreadHandle thread_handle,
            LogBlockReaderInfo &log_block_reader_info) {
    LOGBLOCK_READRETSTATUS read_status;
    auto ret = GetNextLogBlockToRead(thread_handle, &log_block_reader_info, 100, &read_status);
    if (ret == 0 && read_status == LOGBLOCK_READRETSTATUS_GET) {
        return true;
    }
    return false;
}

bool LogBlockConsumer::OpenLogBlockThreadHandle(LogBlockReaderThreadHandle &reader_thread_handle) {
    auto ret = OpenLogBlockReaderThreadHandle(&reader_thread_handle);
    if (ret == 0) {
        return true;
    }
    return false;
}

void LogBlockConsumer::Stop() {
    running_.store(false);
}

} // namespace logblock
} // namespace netaos
} // namespace hozon
