/*
 * @Author: your name
 * @Date: 2022-01-24 11:05:41
 * @LastEditTime: 2022-02-22 10:47:38
 * @LastEditors: Please set LastEditors
 * @Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 * @FilePath: /mdc_sdk/home/tjy/workspace/gitlab/master/ap/modules/canstack/hz_chassis/include/data_type.h
 */
#pragma once

enum ChassisDataType {
    CHASSIS_NONE = 0,
    CHASSIS_INFO = 1,
    CHASSIS_FLC_INFO,
    CHASSIS_FLR_INFO,
    CHASSIS_APA_INFO,
    CHASSIS_PARKING_NUMBER,
    CHASSIS_E3,
    MCU_TO_EGO,
    MCU_TO_STATEMACHINE,
};

