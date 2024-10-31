#pragma once

#include "framework/log_interface/common.h"
#include "framework/log_interface/log_message.h"
#include "framework/log_interface/log_module.h"

/*****************************************************************************/
/* all netaos logger entrance                                                */
/*****************************************************************************/
#define LOG_INTERFACE(module, level)                              \
  if ((int)netaos::framework::loginterface::LogModule::Instance() \
          ->get_level_by_module(#module) <= level)                \
  LOG_INTERNAL(#module, level)

/*****************************************************************************/
/* used by main entrance for :                                               */
/*   1. normal interface                                                     */
/*   2. conditional log interface                                            */
/* if module is not exist, create a new module                               */
/*****************************************************************************/
#define LOG_INTERNAL(module, level)                                       \
  LOG_MESSAGE_##level(                                                    \
      __FILE__, __LINE__,                                                 \
      netaos::framework::loginterface::LogModule::Instance()->get_logger( \
          module))                                                        \
      .stream()

#define LOG_MESSAGE_DEBUG netaos::framework::loginterface::LogMessageDebug
#define LOG_MESSAGE_INFO netaos::framework::loginterface::LogMessageInfo
#define LOG_MESSAGE_WARNING netaos::framework::loginterface::LogMessageWarning
#define LOG_MESSAGE_ERROR netaos::framework::loginterface::LogMessageError
#define LOG_MESSAGE_FATAL netaos::framework::loginterface::LogMessageFatal
