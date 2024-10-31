/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
 * Description:
 */

#ifndef SRC_PLOG_SRC_STAT_INFOHANDLE_H
#define SRC_PLOG_SRC_STAT_INFOHANDLE_H

#include <string>
#include <memory>

#include "StatReturnCode.h"
#include "InfoWriter.h"

namespace rbs {
namespace plog {
namespace stat {
class InfoHandleImpl;

/**
 * @brief A handle of A stat object, which is a file in /tmp/__PLOG_STAT_${handleName}.${PID}, user can create
 *        InfoWriter from this to write stat info str into it. the file will be removed if the process exit normally,
 *        every time the Init is called, we will check in the tmp to find PLOG_STAT file and remove the zombie ones,
 *        whose process owner is closed unexpectedly
 */
class InfoHandle {
public:
    /**
     * @brief A Constructor, but user shall not use this, user should use CreateHandleInstance to create handle
     */
    explicit InfoHandle(std::shared_ptr<InfoHandleImpl> impl);

    struct HandleCreateResult {
        StatReturnCode res;
        /** in every case, if handle is not nullptr, then it can be used */
        std::shared_ptr<InfoHandle> handle;
    };
    /**
     * @brief create a handle instance
     * @param handleName the name of this handle, this will be performed at the file name, which should be
     *        '__PLOG_STAT_${handleName}.${PID}', the max len is 64, the over sized part will be truncated
     * @param blockSize the max char size of each INfoWriter can write, including the ending '\0', the uplimit is 10k
     * @return struct HandleCreateResult, the code will indicate the result, which is standard return code, user can use
     *         Code2Str(StatReturnCode code) to print the related info string. if the create successed, the handle will
     *         not be nullptr
     * @retval OVER_SIZE_NAME if the handleName is truncated, but handle can still be used
     * @retval DUPLICATE_HANDLE the handleName you has used, this error is higher than OVER_SIZE_NAME, in this case
     *         we will try to add prefix after the handleName, the file name will be something like :
     *         __PLOG_STAT_${handleName}_01.${PID}, the limit is __PLOG_STAT_${handleName}_99.${PID}, then it will fail
     */
    static HandleCreateResult CreateHandleInstance(const std::string& handleName, uint64_t blockSize);

    struct WriterCreateResult {
        StatReturnCode res;
        /** in every case, if the writer is not nullptr, then it can be used */
        std::shared_ptr<InfoWriter> writer;
    };
    /**
     * @brief Generate a child InfoWriter related to this InfoHandle
     * @param keyStr the key string of the InfoWriter, the max len is 128 (including the ending '\0'), the over size
     *        part will be truncated, and the OVER_SIZE_NAME will be returned, but it can still be used
     * @return res = standard return code, user can use Code2Str(StatReturnCode code) to print the related info string
     * @return writer = the created writer, which will be null, if some error happened
     * @retval OUT_OF_RESOURCE for one handle, you can create at most 1024 writer, otherwise this errcode will be
     *         returned, the 1024 means all writers created in this handle's lifetime
     * @note if you input the same keyStr (including after truncated strs), we will give you the same instance last we
     *       created for you, and the res.res will be DUPLICATE_KEY (this is higher than OVER_SIZE_NAME)
     */
    WriterCreateResult CreateWriter(const std::string& keyStr);

    /**
     * @brief The destructor, where the handle will actually closed, when all child InfoWriters are dead, the handle
     *        will finally closed, where the related file will be deleted
     */
    ~InfoHandle() = default;

private:
    std::shared_ptr<InfoHandleImpl> impl_;
};
}
}
}

#endif // SRC_PLOG_SRC_STAT_INFOHANDLE_H
