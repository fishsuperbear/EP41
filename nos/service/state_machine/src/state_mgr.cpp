#include <chrono>
#include <cfloat>
#include "map_manage.h"
#include "state_mgr.h"
#include "state/cruising.h"
#include "state/localization.h"
#include "state/map_building.h"
#include "state/parking.h"
#include "state/standby.h"
#include "state/reset.h"


#define START_POINT_DISTANCE (80.0)
#define TOLERANCE (1e-9)
#define INVALID_MAP_ID (0)

StateManager::StateManager()
    : parking_state_(PARKING_STATE::FAPA_PARKING_IN)
    , standby_state_(STANDBY_STATE::STANDBY_FAPA_PARKING_IN)
    , reset_(nullptr)
    , th_(&StateManager::MapSearch, this)
    , map_id_(INVALID_MAP_ID)
    , stop_(false)
{
}

StateManager::~StateManager()
{
    stop_.store(true);
    if (th_.joinable()) {
        th_.join();
    }
}

void StateManager::MapSearch()
{
    while (!stop_.load()) {
        if (G_In_Imu() && G_In_SM_Per()) {
            if (G_In_SM_Per()->hpp_perception_status.HPA_GuideSts == 0x02) { // 室外地面
                NODE_LOG_WARN << "StateManager HPA_GuideSts=" << G_In_SM_Per()->hpp_perception_status.HPA_GuideSts;
                double min_dis = DBL_MAX;
                int min_id = INVALID_MAP_ID;
                if ((G_In_Imu()->ins_info.sysStatus == 1) || ((G_In_Imu()->ins_info.sysStatus == 2) &&
                    (G_In_Imu()->ins_info.gpsStatus == 4 || G_In_Imu()->ins_info.gpsStatus == 5 ||
                     G_In_Imu()->ins_info.gpsStatus == 6 || G_In_Imu()->ins_info.gpsStatus == 7 ||
                     G_In_Imu()->ins_info.gpsStatus == 8 || G_In_Imu()->ins_info.gpsStatus == 9))) {
                    for (auto id : hozon::netaos::MapManage::getInstance().pollAllMap()) {
                        auto xy = hozon::netaos::MapManage::getInstance().getXY(id);
                        NODE_LOG_WARN << "StateManager latitude=" << G_In_Imu()->ins_info.latitude;
                        NODE_LOG_WARN << "StateManager longitude=" << G_In_Imu()->ins_info.longitude;
                        double dis = hozon::netaos::MapManage::getInstance().calculateDistance(
                            G_In_Imu()->ins_info.latitude, G_In_Imu()->ins_info.longitude, xy.second, xy.first);
                        NODE_LOG_WARN << "StateManager id=" << id;
                        NODE_LOG_WARN << "StateManager calculate distance=" << dis;

                        if (dis <= START_POINT_DISTANCE && dis < min_dis) {
                            NODE_LOG_WARN << "******** StateManager minid ********" << id;
                            min_dis = dis;
                            min_id = id;
                        }
                    }
                } else {
                    NODE_LOG_ERROR << "StateManager invalid gps";
                }

                NODE_LOG_WARN << "StateManager min_id=" << min_id;
                map_id_.store(min_id);
                hozon::netaos::MapManage::getInstance().setMapId(min_id);
            }
        }
        NODE_LOG_WARN << "=============== StateManager before sleep id ========" << map_id_.load();

        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
}

int32_t StateManager::MapId()
{
    return map_id_.load();
}

void StateManager::Init()
{
    localization_ = std::make_shared<Localization>();
    localization_->Register(this);
    map_building_ = std::make_shared<MapBuilding>();
    map_building_->Register(this);
    ntp_cruising_ = std::make_shared<Cruising>();
    ntp_cruising_->Register(this);

    standby_ = std::make_shared<StandbyFapaParkingIn>();
    standby_->Register(this);
    parking_ = std::make_shared<FapaParkingIn>();
    parking_->Register(this);

    reset_ = std::make_shared<Reset>();
    reset_->Register(this);
}

void StateManager::ExecState()
{
    if (G_ParkingTask()->now_state == Parking_Work_Status::Standby) {
        //
    } else if (G_ParkingTask()->now_state == Parking_Work_Status::Parking) {
        //
    } else if (G_ParkingTask()->now_state == Parking_Work_Status::Localization) {
        NODE_LOG_WARN << "StateManager Transfer State To Localization.";
        localization_->Process();
    } else if (G_ParkingTask()->now_state == Parking_Work_Status::Map_Building) {
        NODE_LOG_WARN << "StateManager Transfer State To Map_Building.";
        map_building_->Process();
    } else if (G_ParkingTask()->now_state == Parking_Work_Status::Cruising_HPP) {
        NODE_LOG_WARN << "StateManager Transfer State To Cruising_HPP.";
        ntp_cruising_->Process();
    } else if (G_ParkingTask()->now_state == Parking_Work_Status::RESET) {
        NODE_LOG_WARN << "StateManager Transfer State To RESET.";
        reset_->Process();
    } else {
        NODE_LOG_ERROR << "----------cur Parking_Work_Status state----------" << static_cast<int>(G_ParkingTask()->now_state);
    }
}

void StateManager::UpdateStandbyState(STANDBY_STATE state)
{
    standby_state_ = STANDBY_STATE::STANDBY_FAPA_PARKING_IN;

    if (G_In_Chassis()->CDCS11_APA_FunctionMode == 0x1) {
        standby_state_ = STANDBY_STATE::STANDBY_FAPA_PARKING_IN;
    } else if (G_In_Chassis()->CDCS11_APA_FunctionMode == 0x2) {
        standby_state_ = STANDBY_STATE::STANDBY_FAPA_PARKING_OUT;
    } else if (G_In_Chassis()->CDCS11_APA_FunctionMode == 0x3) {
        standby_state_ = STANDBY_STATE::STANDBY_FAPA_PARKING_SELECT;
    } else {

    }

    if (G_In_Chassis()->BTM1_FunctionMode == 0x1) {
        standby_state_ = STANDBY_STATE::STANDBY_RPA_PARKING_IN;
    } else if (G_In_Chassis()->BTM1_FunctionMode == 0x2) {
        standby_state_ = STANDBY_STATE::STANDBY_RPA_PARKING_OUT;
    } else if (G_In_Chassis()->BTM1_FunctionMode == 0x3) {
        standby_state_ = STANDBY_STATE::STANDBY_RPA_PARKING_SELECT;
    } else if (G_In_Chassis()->BTM1_FunctionMode == 0x4) {
        standby_state_ = STANDBY_STATE::STANDBY_RPA_PARKING_LINE;
    } else {

    }

    StandByStateTransition();
}

void StateManager::StandByStateTransition()
{
    switch (standby_state_)
    {
    case STANDBY_STATE::STANDBY_FAPA_PARKING_IN:
        Create<StandbyFapaParkingIn>(standby_);
        break;
    case STANDBY_STATE::STANDBY_FAPA_PARKING_OUT:
        /* code */
        break;
    case STANDBY_STATE::STANDBY_FAPA_PARKING_SELECT:
        /* code */
        break;
    case STANDBY_STATE::STANDBY_RPA_PARKING_IN:
        /* code */
        break;
    case STANDBY_STATE::STANDBY_RPA_PARKING_OUT:
        /* code */
        break;
    case STANDBY_STATE::STANDBY_RPA_PARKING_SELECT:
        /* code */
        break;
    case STANDBY_STATE::STANDBY_RPA_PARKING_LINE:
        /* code */
        break;
    case STANDBY_STATE::STANDBY_DEFAULT:
        /* code */
        break;

    default:
        break;
    }
}

void StateManager::UpdateParkingState(PARKING_STATE state)
{
    parking_state_ = state;
    ParkingStateTransition();
}

void StateManager::ParkingStateTransition()
{
    switch (parking_state_)
    {
    case PARKING_STATE::FAPA_PARKING_IN:
        Create<FapaParkingIn>(parking_);
        break;
    case PARKING_STATE::FAPA_PARKING_OUT:
        /* code */
        break;
    case PARKING_STATE::RPA_PARKING_IN:
        /* code */
        break;
    case PARKING_STATE::RPA_PARKING_OUT:
        /* code */
        break;
    case PARKING_STATE::RPA_PARKING_SELECT:
        /* code */
        break;
    case PARKING_STATE::RPA_PARKING_LINE:
        /* code */
        break;
    case PARKING_STATE::TBA_PARKING:
        /* code */
        break;
    case PARKING_STATE::PARKING:
        /* code */
        break;

    default:
        break;
    }
}
