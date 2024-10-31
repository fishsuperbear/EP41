#include <signal.h>
#include <unistd.h>

#include <chrono>
#include <condition_variable>
#include <csignal>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "cm/include/proto_cm_writer.h"
#include "log/include/default_logger.h"

#include "proto/localization/localization.pb.h"
#include "proto/perception/perception_obstacle.pb.h"
#include "proto/planning/planning.pb.h"
sig_atomic_t g_stopFlag = 1;
std::condition_variable g_term_cv;
std::mutex g_term_mutex;
int ret;
std::string mode;
hozon::netaos::cm::ProtoCMWriter<hozon::perception::LaneMarkers> lanewriter;
hozon::netaos::cm::ProtoCMWriter<hozon::localization::Localization> locationwriter;
hozon::netaos::cm::ProtoCMWriter<hozon::perception::PerceptionObstacles> objectwriter;
hozon::netaos::cm::ProtoCMWriter<hozon::planning::ADCTrajectory> planningwriter;

void INTSigHandler(int32_t num) {
    static_cast<void>(num);
    std::lock_guard<std::mutex> lk(g_term_mutex);
    g_stopFlag = 0;
    g_term_cv.notify_all();
    DF_LOG_ERROR << "Signal Interactive attention received.";
}
void SendPlanningDataThread() {
    pthread_setname_np(pthread_self(), "SendPlanningDataThread");
    uint32_t count = 0;
    while (g_stopFlag) {
        hozon::planning::ADCTrajectory msg;
        struct timespec time;
        if (0 != clock_gettime(CLOCK_REALTIME, &time)) {
            DF_LOG_ERROR << "clock_gettime fail ";
        }
        // msg.mutable_header()->set_timestamp_sec(static_cast<double>(time.tv_sec) + static_cast<double>(time.tv_nsec) / 1e9);
        // msg.mutable_header()->set_module_name("Localization");
        // msg.mutable_header()->set_sequence_num(count++);
        // msg.mutable_header()->set_lidar_timestamp(4);
        // msg.mutable_header()->set_camera_timestamp(5);
        // msg.mutable_header()->set_radar_timestamp(6);
        // msg.mutable_header()->set_version(7);
        msg.set_total_path_length(1.1);
        msg.set_total_path_time(2.1);
        msg.clear_trajectory_point();
        for (size_t size = 0; size < 1; size++) {
            hozon::common::TrajectoryPoint* trajectory_point = msg.add_trajectory_point();
            trajectory_point->mutable_path_point()->set_x(3.1);
            trajectory_point->mutable_path_point()->set_y(4.1);
            trajectory_point->mutable_path_point()->set_z(5.1);
            trajectory_point->mutable_path_point()->set_theta(6.1);
            trajectory_point->mutable_path_point()->set_kappa(7.1);
            trajectory_point->mutable_path_point()->set_s(8.1);
            trajectory_point->mutable_path_point()->set_l(9.1);
            trajectory_point->mutable_path_point()->set_dkappa(10.1);
            trajectory_point->mutable_path_point()->set_ddkappa(11.1);
            trajectory_point->mutable_path_point()->set_lane_id("12.1");
            trajectory_point->mutable_path_point()->set_x_derivative(13.1);
            trajectory_point->mutable_path_point()->set_y_derivative(14.1);
            trajectory_point->set_v(15.1);
            trajectory_point->set_a(16.1);
            trajectory_point->set_relative_time(17.1);
            trajectory_point->set_da(18.1);
            trajectory_point->set_steer(19.1);
            trajectory_point->mutable_gaussian_info()->set_sigma_x(20.1);
            trajectory_point->mutable_gaussian_info()->set_sigma_y(21.1);
            trajectory_point->mutable_gaussian_info()->set_correlation(22.1);
            trajectory_point->mutable_gaussian_info()->set_area_probability(23.1);
            trajectory_point->mutable_gaussian_info()->set_ellipse_a(24.1);
            trajectory_point->mutable_gaussian_info()->set_ellipse_b(25.1);
            trajectory_point->mutable_gaussian_info()->set_theta_a(26.1);
        }
        hozon::planning_internal::AstarNodeRow* astar_node_rows = msg.mutable_debug()->mutable_astar_decider_info()->add_astar_node_rows();
        astar_node_rows->set_row_index(111);
        ret = planningwriter.Write(msg);
        if (ret < 0) {
            DF_LOG_ERROR << "Fail to planningwriter  " << ret;
        }
        DF_LOG_INFO << "planningwriter   " << count;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void SendLocationDataThread() {
    pthread_setname_np(pthread_self(), "SendLocationDataThread");
    uint32_t count = 0;
    while (g_stopFlag) {
        hozon::localization::Localization msg;
        struct timespec time;
        if (0 != clock_gettime(CLOCK_REALTIME, &time)) {
            DF_LOG_ERROR << "clock_gettime fail ";
        }
        // msg.mutable_header()->set_timestamp_sec(static_cast<double>(time.tv_sec) + static_cast<double>(time.tv_nsec) / 1e9);
        // msg.mutable_header()->set_module_name("Localization");
        // msg.mutable_header()->set_sequence_num(count++);
        // msg.mutable_header()->set_lidar_timestamp(4);
        // msg.mutable_header()->set_camera_timestamp(5);
        // msg.mutable_header()->set_radar_timestamp(6);
        // msg.mutable_header()->set_version(7);
        msg.mutable_header()->mutable_status()->set_error_code((hozon::common::ErrorCode::OK));
        msg.mutable_header()->mutable_status()->set_msg("9");
        msg.mutable_header()->set_frame_id("0.3");
        msg.mutable_pose()->mutable_linear_acceleration_raw_vrf()->set_z(111.1);
        ret = locationwriter.Write(msg);
        if (ret < 0) {
            DF_LOG_ERROR << "Fail to locationwriter  " << ret;
        }
        DF_LOG_INFO << "locationwriter   " << count;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

void SendObjectThread() {
    pthread_setname_np(pthread_self(), "SendObjectThread");
    uint32_t count = 0;
    while (g_stopFlag) {
        hozon::perception::PerceptionObstacles msg;
        struct timespec time;
        if (0 != clock_gettime(CLOCK_REALTIME, &time)) {
            DF_LOG_ERROR << "clock_gettime fail ";
        }
        // msg.mutable_header()->set_timestamp_sec(static_cast<double>(time.tv_sec) + static_cast<double>(time.tv_nsec) / 1e9);
        // msg.mutable_header()->set_module_name("Localization");
        // msg.mutable_header()->set_sequence_num(count++);
        // msg.mutable_header()->set_lidar_timestamp(4);
        // msg.mutable_header()->set_camera_timestamp(5);
        // msg.mutable_header()->set_radar_timestamp(6);
        // msg.mutable_header()->set_version(7);
        msg.mutable_header()->mutable_status()->set_error_code((hozon::common::ErrorCode::OK));
        msg.mutable_header()->mutable_status()->set_msg("9");
        msg.mutable_header()->set_frame_id("0.3");
        msg.mutable_frame_to_vehicle_pose()->mutable_point3d()->set_z(111.11);
        ret = objectwriter.Write(msg);
        if (ret < 0) {
            DF_LOG_ERROR << "Fail to objectwriter  " << ret;
        }
        DF_LOG_INFO << "objectwriter   " << count;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}
void SendLaneDataThread() {
    pthread_setname_np(pthread_self(), "SendLaneDataThread");
    uint32_t count = 0;
    while (g_stopFlag) {
        hozon::perception::LaneMarkers msg;
        struct timespec time;
        if (0 != clock_gettime(CLOCK_REALTIME, &time)) {
            DF_LOG_ERROR << "clock_gettime fail ";
        }
        // msg.mutable_header()->set_timestamp_sec(static_cast<double>(time.tv_sec) + static_cast<double>(time.tv_nsec) / 1e9);
        // msg.mutable_header()->set_module_name("LaneMarkers");
        // msg.mutable_header()->set_sequence_num(count++);
        // msg.mutable_header()->set_lidar_timestamp(4);
        // msg.mutable_header()->set_camera_timestamp(5);
        // msg.mutable_header()->set_radar_timestamp(6);
        // msg.mutable_header()->set_version(7);
        msg.mutable_header()->mutable_status()->set_error_code((hozon::common::ErrorCode::OK));
        msg.mutable_header()->mutable_status()->set_msg("9");
        msg.mutable_header()->set_frame_id("0.3");
        msg.set_lane_count(111);
        ret = lanewriter.Write(msg);
        if (ret < 0) {
            DF_LOG_ERROR << "Fail to lanewriter  " << ret;
        }
        DF_LOG_INFO << "lanewriter   " << count;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}
int main(int argc, char* argv[]) {
    mode = std::string(argv[1]);
    DefaultLogger::GetInstance().InitLogger();
    if (mode == "1") {
        int32_t ret = lanewriter.Init(0, "npp_lane");
        if (ret < 0) {
            DF_LOG_ERROR << "Fail to init lanewriter " << ret;
            return -1;
        }
        ret = locationwriter.Init(0, "nnp_localization");
        if (ret < 0) {
            DF_LOG_ERROR << "Fail to init locationwriter " << ret;
            return -1;
        }
        ret = objectwriter.Init(0, "ObjectFusion1");
        if (ret < 0) {
            DF_LOG_ERROR << "Fail to init objectwriter " << ret;
            return -1;
        }
    } else if (mode == "2") {
        int32_t ret = lanewriter.Init(0, "hpp_lane");
        if (ret < 0) {
            DF_LOG_ERROR << "Fail to init lanewriter " << ret;
            return -1;
        }
        ret = locationwriter.Init(0, "/perception/parking/slam_location");
        if (ret < 0) {
            DF_LOG_ERROR << "Fail to init locationwriter " << ret;
            return -1;
        }
        ret = objectwriter.Init(0, "/perception/parking/fusion_object");
        if (ret < 0) {
            DF_LOG_ERROR << "Fail to init objectwriter " << ret;
            return -1;
        }
    }
    ret = planningwriter.Init(0, "ego_trajectory");
    if (ret < 0) {
        DF_LOG_ERROR << "Fail to init planningwriter " << ret;
        return -1;
    }
    std::thread planningAct(SendPlanningDataThread);
    planningAct.detach();
    std::thread locationAct(SendLocationDataThread);
    locationAct.detach();
    std::thread laneDataAct(SendLaneDataThread);
    laneDataAct.detach();
    std::thread objectAct(SendObjectThread);
    objectAct.detach();
    {
        std::unique_lock<std::mutex> lck(g_term_mutex);
        while (g_stopFlag) {
            g_term_cv.wait(lck);
        }
    }
    lanewriter.Deinit();
    locationwriter.Deinit();
    objectwriter.Deinit();
    planningwriter.Deinit();
    DF_LOG_INFO << "Deinit end." << ret;
}