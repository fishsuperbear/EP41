/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 */

#ifndef DDS_CORE_POLICY_DATA_PROCESS_TYPE_HPP
#define DDS_CORE_POLICY_DATA_PROCESS_TYPE_HPP

#include <RT-DDS/dds/core/policy/DataProcessTypeKind.hpp>

namespace dds {
namespace core {
namespace policy {
/**
 * @defgroup DIRECT_RETURN DIRECT_RETURN_POLICY
 * @brief Specify whether the user wish to let dds directly return data using OnDataProcess
 * or use the standard OnDataAvailable which is the default option
 * @ingroup DIRECT_RETURN
 */
class DataProcessType {
public:
    DataProcessType() = default;

    ~DataProcessType() = default;

    static DataProcessType DirectDataProcess() noexcept
    {
        return DataProcessType{DataProcessTypeKind::DIRECT_DATA_PROCESS};
    }

    static DataProcessType NormalTake() noexcept
    {
        return DataProcessType{DataProcessTypeKind::NORMAL_TAKE};
    }
    /**
     * @brief creates a directDataProcess element
     * @param flag the bool value showing if direct return should be used
     */
    explicit DataProcessType(DataProcessTypeKind kind) noexcept : dataProcessTypeKind_(kind)
    {}

    /**
     * @ingroup DIRECT_RETURN
     * @brief allow the user to set whether to use direct return or not
     * @par Description
     * 1. set the value that signals whether the user wish to use direct return
     * @param flag the bool value showing if direct return should be used
     * @return void
     */
    void SetDataProcessType(DataProcessTypeKind kind) noexcept
    {
        dataProcessTypeKind_ = kind;
    }

    /**
     * @ingroup DIRECT_RETURN
     * @brief allow the user to get whether to use direct return or not
     * @par Description
     * 1. get the value that signals whether the user wish to use direct return
     * @param NONE
     * @return DataProcessTypeKind indicating which type of callback to use
     */
    DataProcessTypeKind GetDataProcessType() const noexcept
    {
        return dataProcessTypeKind_;
    }

    /**
     * @ingroup DIRECT_RETURN
     * @brief allow the user to get if current policy is set to direct data process
     * @param NONE
     * @return bool indicating whether direct data process is used
     */
    bool IsDirectDataProcess() const noexcept
    {
        if (dataProcessTypeKind_ == DataProcessTypeKind::DIRECT_DATA_PROCESS) {
            return true;
        } else {
            return false;
        }
    }
private:
    DataProcessTypeKind dataProcessTypeKind_ {DataProcessTypeKind::NORMAL_TAKE};
};
}
}
}

#endif // DDS_CORE_POLICY_DATA_PROCESS_TYPE_HPP
