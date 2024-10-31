
#include "proxy_mcu2ego.h"

#include <memory>

#include "logger.h"
#include "common.h"

namespace hozon {
namespace netaos {
namespace sensor {
template <typename DataType>
static DataType Int2ProtoEnum(const DataType default_value, const int number) {
    DataType enum_data;
    const auto descriptor = google::protobuf::GetEnumDescriptor<DataType>()->FindValueByNumber(number);
    if (descriptor == nullptr || !google::protobuf::internal::ParseNamedEnum(descriptor->type(), descriptor->name(), &enum_data)) {
        enum_data = default_value;
    }
    return enum_data;
}

Mcu2EgoProxy::Mcu2EgoProxy() : _mcu2ego_pub_last_seq(0) {}

std::shared_ptr<hozon::functionmanager::FunctionManagerIn> Mcu2EgoProxy::Trans(ara::com::SamplePtr<::hozon::netaos::AlgMcuToEgoFrame const> data) {
    std::shared_ptr<hozon::functionmanager::FunctionManagerIn> mcu2funcMgr = std::make_shared<hozon::functionmanager::FunctionManagerIn>();

    if (mcu2funcMgr == nullptr) {
        SENSOR_LOG_ERROR << "mcu2funcMgr Allocate got nullptr!";
        return nullptr;
    }
    // receive someip mcu2funcMgr struct data to mcu2funcMgr proto
    auto pb_header = mcu2funcMgr->mutable_header();
    pb_header->set_seq(data->header.seq);
    pb_header->set_frame_id(std::string(data->header.frameId.begin(), data->header.frameId.end()));

    struct timespec time;
    if (0 != clock_gettime(CLOCK_REALTIME, &time)) {
        SENSOR_LOG_ERROR << "clock_gettime fail ";
    }

    // data->header.stamp.sec = time.tv_sec;
    // data->header.stamp.nsec = time.tv_nsec;

    pb_header->set_publish_stamp(static_cast<double>(time.tv_sec) + static_cast<double>(time.tv_nsec) / 1000 / 1000 / 1000);
    pb_header->set_gnss_stamp(data->header.gnssStamp.sec + data->header.gnssStamp.nsec / (1000.0 * 1000.0 * 1000.0));

    mcu2funcMgr->set_driver_mode(Int2ProtoEnum(hozon::functionmanager::DriveMode::NONE, data->drive_mode));
    mcu2funcMgr->set_fct_2_soc_tbd_u32_01(data->FCT2SOC_TBD_u32_01);
    mcu2funcMgr->set_fct_2_soc_tbd_u32_02(data->FCT2SOC_TBD_u32_02);
    mcu2funcMgr->set_fct_2_soc_tbd_u32_03(data->FCT2SOC_TBD_u32_03);
    mcu2funcMgr->set_fct_2_soc_tbd_u32_04(data->FCT2SOC_TBD_u32_04);
    mcu2funcMgr->set_fct_2_soc_tbd_u32_05(data->FCT2SOC_TBD_u32_05);

    mcu2funcMgr->set_ta_pilot_mode(Int2ProtoEnum(hozon::functionmanager::TaPilotMode::NO_CONTROL, data->ta_pilot_mode));
    // nnp input
    mcu2funcMgr->mutable_fct_nnp_in()->set_longitud_ctrl_dectostop_req(data->msg_mcu_nnp.LongitudCtrlDecToStopReq);
    mcu2funcMgr->mutable_fct_nnp_in()->set_longitud_ctrl_driveoff(data->msg_mcu_nnp.LongitudCtrlDriveOff);
    mcu2funcMgr->mutable_fct_nnp_in()->set_driveoff_inhibition(data->msg_mcu_nnp.DriveOffinhibition);
    mcu2funcMgr->mutable_fct_nnp_in()->set_driveoff_inhibition_objtype(Int2ProtoEnum(hozon::functionmanager::DriveOffInhibitionObjType::UNKNOWN, data->msg_mcu_nnp.DriveOffinhibitionObjType));
    mcu2funcMgr->mutable_fct_nnp_in()->set_lcsndconfirm(Int2ProtoEnum(hozon::functionmanager::LcsndConfirm::NO_ACTION_LCSND, data->msg_mcu_nnp.Lcsndconfirm));
    mcu2funcMgr->mutable_fct_nnp_in()->set_turnlight_reqst(Int2ProtoEnum(hozon::functionmanager::TurnLightReq::NO_REQ_LIGHT, data->msg_mcu_nnp.TurnLightReqSt));
    mcu2funcMgr->mutable_fct_nnp_in()->set_lcsnd_request(Int2ProtoEnum(hozon::functionmanager::LcsndRequest::NO_ACTION_LCDND_REQ, data->msg_mcu_nnp.Lcsndrequest));
    mcu2funcMgr->mutable_fct_nnp_in()->set_paymode_confirm(data->msg_mcu_nnp.PayModeConfirm);
    mcu2funcMgr->mutable_fct_nnp_in()->set_spd_adapt_comfirm(data->msg_mcu_nnp.SpdAdaptComfirm);
    mcu2funcMgr->mutable_fct_nnp_in()->set_alc_mode(Int2ProtoEnum(hozon::functionmanager::AlcMode::NEED_CONFIRM_ALC, data->msg_mcu_nnp.ALC_mode));
    mcu2funcMgr->mutable_fct_nnp_in()->set_driving_mode(Int2ProtoEnum(hozon::functionmanager::DrivingMode::UNKNOWN_MODE, data->msg_mcu_nnp.ADSDriving_mode));
    mcu2funcMgr->mutable_fct_nnp_in()->set_longitud_ctrl_setspeed(data->msg_mcu_nnp.longitudCtrlSetSpeed);
    mcu2funcMgr->mutable_fct_nnp_in()->set_longitud_ctrl_setdistance(data->msg_mcu_nnp.longitudCtrlSetDistance);
    mcu2funcMgr->mutable_fct_nnp_in()->set_nnp_sysstate(Int2ProtoEnum(hozon::functionmanager::NNPSysState::NNPS_OFF, data->msg_mcu_nnp.NNPSysState));
    mcu2funcMgr->mutable_fct_nnp_in()->mutable_light_signal_state()->set_hazardlampst(data->msg_mcu_nnp.HazardLampSt);
    mcu2funcMgr->mutable_fct_nnp_in()->mutable_light_signal_state()->set_highbeamst(data->msg_mcu_nnp.HighBeamSt);
    mcu2funcMgr->mutable_fct_nnp_in()->mutable_light_signal_state()->set_hornst(data->msg_mcu_nnp.HornSt);
    mcu2funcMgr->mutable_fct_nnp_in()->mutable_light_signal_state()->set_lowbeamst(data->msg_mcu_nnp.LowBeamSt);
    mcu2funcMgr->mutable_fct_nnp_in()->mutable_light_signal_state()->set_lowhighbeamst(data->msg_mcu_nnp.LowHighBeamSt);
    mcu2funcMgr->mutable_fct_nnp_in()->set_acc_target_id(data->msg_mcu_nnp.acc_target_id);
    mcu2funcMgr->mutable_fct_nnp_in()->set_alc_warning_state(data->msg_mcu_nnp.alc_warnning_state);
    mcu2funcMgr->mutable_fct_nnp_in()->set_alc_warning_target_id(data->msg_mcu_nnp.alc_warnning_target_id);
    uint8_t nnp_ori_state = data->FCT2SOC_TBD_u32_02 & 0xFF;
    mcu2funcMgr->mutable_fct_nnp_in()->set_nnp_original_state(Int2ProtoEnum(hozon::functionmanager::NNPSysState::NNPS_OFF, nnp_ori_state));
    uint8_t acc_state = (data->FCT2SOC_TBD_u32_02 >> 9) & 0xFF;
    mcu2funcMgr->mutable_fct_nnp_in()->set_acc_state(Int2ProtoEnum(hozon::functionmanager::FctToNnpInput::ACC_OFF, acc_state));
    uint8_t npilot_state = (data->FCT2SOC_TBD_u32_02 >> 17) & 0xFF;
    mcu2funcMgr->mutable_fct_nnp_in()->set_npilot_state(Int2ProtoEnum(hozon::functionmanager::FctToNnpInput::PILOT_OFF, npilot_state));
    mcu2funcMgr->mutable_fct_avp_in()->set_sys_command(Int2ProtoEnum(hozon::functionmanager::AvpFctIn_SysCmdType_NOCMDTYPE, data->msg_mcu_avp.system_command));
    mcu2funcMgr->mutable_fct_avp_in()->set_sys_mode(Int2ProtoEnum(hozon::functionmanager::AvpFctIn_StateType_NOSTTYPE, data->msg_mcu_avp.AVPSysMode));
    mcu2funcMgr->mutable_fct_avp_in()->set_sys_run_state(Int2ProtoEnum(hozon::functionmanager::AvpFctIn_SysRunState_STOP, data->msg_mcu_avp.avp_run_state));
    mcu2funcMgr->mutable_fct_avp_in()->set_sys_warning_info(Int2ProtoEnum(hozon::functionmanager::AvpFctIn_WarningInfoErrorType_NO_ERROR, data->msg_mcu_avp.pnc_warninginfo));

    if (mcu2funcMgr->fct_nnp_in().npilot_state() == hozon::functionmanager::FctToNnpInput::PILOT_ACTIVE ||
        mcu2funcMgr->fct_nnp_in().npilot_state() == hozon::functionmanager::FctToNnpInput::PILOT_SUSPEND) {
        mcu2funcMgr->set_adas_mode(hozon::functionmanager::PILOT);
    } else {
        if (mcu2funcMgr->fct_nnp_in().acc_state() == hozon::functionmanager::FctToNnpInput::ACC_ACTIVE ||
            mcu2funcMgr->fct_nnp_in().acc_state() == hozon::functionmanager::FctToNnpInput::ACC_OVERRIDE) {
            mcu2funcMgr->set_adas_mode(hozon::functionmanager::ACC);
        } else {
            mcu2funcMgr->set_adas_mode(hozon::functionmanager::NO_ADAS_MODE);
        }
    }

    mcu2funcMgr->mutable_fct_avp_in()->set_sys_mode(Int2ProtoEnum(hozon::functionmanager::AvpFctIn::NOSTTYPE, data->msg_mcu_avp.AVPSysMode));
    mcu2funcMgr->mutable_fct_avp_in()->set_sys_command(Int2ProtoEnum(hozon::functionmanager::AvpFctIn::NOCMDTYPE, data->msg_mcu_avp.system_command));
    mcu2funcMgr->mutable_fct_avp_in()->set_sys_run_state(Int2ProtoEnum(hozon::functionmanager::AvpFctIn::STOP, data->msg_mcu_avp.avp_run_state));
    mcu2funcMgr->mutable_fct_avp_in()->set_sys_warning_info(Int2ProtoEnum(hozon::functionmanager::AvpFctIn::NO_ERROR, data->msg_mcu_avp.pnc_warninginfo));

    if(_mcu2ego_pub_last_seq && (mcu2funcMgr->mutable_header()->seq() > _mcu2ego_pub_last_seq) 
        && ((mcu2funcMgr->mutable_header()->seq() - _mcu2ego_pub_last_seq) != 1)) {
        SENSOR_LOG_WARN << "imu ins info lost data. receive seq: " << mcu2funcMgr->mutable_header()->seq() \
            << " last seq : "  << _mcu2ego_pub_last_seq  \
            << " seq diff : " << (mcu2funcMgr->mutable_header()->seq() - _mcu2ego_pub_last_seq)
            << " interval : " << (GetAbslTimestamp() \
                - mcu2funcMgr->mutable_header()->gnss_stamp()) << " s";
    } else if ((GetAbslTimestamp() 
            - mcu2funcMgr->mutable_header()->gnss_stamp()) > 0.1f) {  // 0.1s
        SENSOR_LOG_WARN << "imu ins info link latency : " << (GetAbslTimestamp() \
                - mcu2funcMgr->mutable_header()->gnss_stamp()) << " s";
    }
    _mcu2ego_pub_last_seq = mcu2funcMgr->mutable_header()->seq();

    return mcu2funcMgr;
}
}  // namespace sensor
}  // namespace netaos
}  // namespace hozon
