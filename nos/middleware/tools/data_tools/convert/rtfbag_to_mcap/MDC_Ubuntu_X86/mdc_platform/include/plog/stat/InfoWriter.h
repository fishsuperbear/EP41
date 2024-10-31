/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
 * Description:
 */

#ifndef SRC_PLOG_SRC_STAT_INFOWRITER_H
#define SRC_PLOG_SRC_STAT_INFOWRITER_H

#include <memory>
#include <string>

#include "StatReturnCode.h"

namespace rbs {
namespace plog {
namespace stat {
class InfoWriterImpl;

/**
 * @brief An operator class for user to write stat info, each InfoWriter represent a stat key, the key str is set in the
 *        InfoHandle::CreateWriter. the InfoWriter actually owns the Handle, only when all InfoWriters are closed, the
 *        file will be closed and removed
 */
class InfoWriter {
public:
    /**
     * @brief The only constructor, which should only be used by InfoHandle
     */
    explicit InfoWriter(std::shared_ptr<InfoWriterImpl> impl);

    /**
     * @brief Start a new info output, the info last written to this writer will loss
     * @return standard return code, user can use Code2Str(StatReturnCode code) to print the related info string
     */
    StatReturnCode BeginNewOutput();

    /**
     * @brief append a str to the output, you must first call BeginNewOutput and success
     * @param theAppendingString the string to output to the related key
     * @return standard return code, user can use Code2Str(StatReturnCode code) to print the related info string
     * @retval SIZE_OVERFLOW the written str has exceed the block size set in the father handle, the oversized part is
     *         truncated
     */
    StatReturnCode Output(const std::string& theAppendingString);

    /**
     * @brief Finish the current output
     * @return standard return code, user can use Code2Str(StatReturnCode code) to print the related info string. If an
     *         error happened, the last output info may lost, but you don't need to call the finish again
     * @note You must call this after your success call to BeginNewOutput
     */
    StatReturnCode FinishCurrentOutput();

    virtual ~InfoWriter();

private:
    std::shared_ptr<InfoWriterImpl> impl_;
};
}
}
}

#endif // SRC_PLOG_SRC_STAT_INFOWRITER_H
