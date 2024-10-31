/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
 *
 * @File: processor_factory.h
 * @Date: 2023/09/07
 * @Author: cheng
 * @Desc: --
 */

#pragma once
#ifndef SERVICE_DATA_COLLECTION_PROCESSOR_INCLUDE_PROCESSOR_FACTORY_H__
#define SERVICE_DATA_COLLECTION_PROCESSOR_INCLUDE_PROCESSOR_FACTORY_H__

#include "impl/copier.h"
#include "impl/compressor.h"
#include "impl/mcap_changer.h"
#include "impl/mcap_h265_rw.h"
#include "impl/add_data.h"
#include "impl/all_file_merge.h"
#include "impl/get_dynamic_config.h"
#include "impl/manager_old_file.h"
#include "impl/desense_manager.h"
#include "processor.h"
#include "utils/include/dc_logger.hpp"

namespace hozon {
namespace netaos {
namespace dc {
enum ProcessorEnum {
    COPIER,
    COMPRESS,
    MCAPDEAL,
    DESENSE,
    ADDDATA,
    ALLFILEMERGE,
    GETCONFIG,
    OLD_FILE_MANAGER,
    DESENSEMANAGER,
};

class ProcessorFactory {
   public:
    static ProcessorFactory* getInstance() {
        static ProcessorFactory cf;
        return &cf;
    }

    Processor* createProcess(ProcessorEnum processorType) {
        switch (processorType) {
            case COPIER: {
                DC_RETURN_NEW(Copier());
            }
            case COMPRESS: {
                DC_RETURN_NEW(Compressor());
            }
            case MCAPDEAL: {
                DC_RETURN_NEW(McapChanger());
            }
            case DESENSE: {
                DC_RETURN_NEW(McapH265RW());
            }
            case ADDDATA: {
                DC_RETURN_NEW(AddData());
            }
            case ALLFILEMERGE: {
                DC_RETURN_NEW(AllFileMerge());
            }
            case GETCONFIG: {
                DC_RETURN_NEW(GetDynamicConfig());
            }
            case OLD_FILE_MANAGER: {
                DC_RETURN_NEW(ManagerOldFiles());
            }
            case DESENSEMANAGER: {
                DC_RETURN_NEW(DesenseManager());
            }
        }
        DC_SERVER_LOG_ERROR << "create processor error: not support type " << processorType;
        return nullptr;
    };

   private:
    ProcessorFactory() {}

    ProcessorFactory(ProcessorFactory&& cf) = delete;

    ~ProcessorFactory() {}

    ProcessorFactory(const ProcessorFactory& cf) = delete;
    ProcessorFactory& operator=(const ProcessorFactory& cf) = delete;
};

}  // namespace dc
}  // namespace netaos
}  // namespace hozon

#endif  // SERVICE_DATA_COLLECTION_PROCESSOR_INCLUDE_PROCESSOR_FACTORY_H__
