

#ifndef INS_PVATB_PUBLISHER_H
#define INS_PVATB_PUBLISHER_H

#include "can_parser_ins_pvatb.h"
#include "publisher.h"
#include "adf/include/node_skeleton.h"
#include "proto/common/geometry.pb.h"

namespace hozon {
namespace netaos {
namespace ins_pvatb {

class InsPvatbPublisher : public hozon::netaos::canstack::Publisher {
   public:
    static InsPvatbPublisher* Instance();
    ~InsPvatbPublisher();

    int Init() override;
    void Pub() override;
    int Stop() override;
    int PubData(int i, unsigned char* data, int size);

   protected:
    // std::shared_ptr<hozon::interface::gnssinfo::skeleton::HozonInterface_GnssInfoSkeleton> skeleton_gnss_info;
    // std::shared_ptr<hozon::interface::imu::skeleton::HozonInterface_ImuInfoSkeleton> skeleton_imu;
    // std::shared_ptr<hozon::interface::ins::skeleton::HozonInterface_InsInfoSkeleton> skeleton_ins;
    std::shared_ptr<hozon::netaos::cm::Skeleton> skeleton_imu_ins_;
    std::shared_ptr<hozon::netaos::cm::Skeleton> skeleton_gnss_info_;

    bool serving_;
    uint32_t gnss_heading_seqid;
    // uint32_t gnss_vel_seqid;
    // uint32_t gnss_pos_seqid;
    uint32_t imu_seqid;
    // uint32_t ins_seqid;

   private:
    InsPvatbPublisher();
    template <typename T1, typename T2>
    void GeometryPoitTransLate(const T1& fromPoint, T2& endPoint);
    hozon::common::Point3D GeometryPoitTransLate(const GeometryPoit& fromPoint);
    double GetRealTimestamp();
    static InsPvatbPublisher* sinstance_;
    uint64_t imu_ins_pub_last_time;
    // uint64_t gnss_pub_last_time;
    hozon::common::Point3D point_3d;
};

}  // namespace ins_pvatb
}  // namespace netaos
}  // namespace hozon

#endif  // #define INS_PVATB_PUBLISHER_H
