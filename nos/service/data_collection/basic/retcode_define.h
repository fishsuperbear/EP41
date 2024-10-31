/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
 *
 * @File: retcode_define.h
 * @Date: 2023/08/16
 * @Author: cheng
 * @Desc: --
 */

#pragma once
#ifndef MIDDLEWARE_TOOLS_DATA_COLLECT_BASIC_RETCODE_DEFINE_H
#define MIDDLEWARE_TOOLS_DATA_COLLECT_BASIC_RETCODE_DEFINE_H


namespace hozon {
namespace netaos {
namespace dc {

enum RetCode {
    SUCCESS = 0,
    LOAD_TRIGGER_CONFIG_FAILED = 100,
};

}  // namespace dc
}  // namespace netaos
}  // namespace hozon

#endif  // MIDDLEWARE_TOOLS_DATA_COLLECT_BASIC_RETCODE_DEFINE_H



