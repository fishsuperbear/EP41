/* -*- mode: C++ -*-
 *
 *  Copyright (C) 2011, 2012 Austin Robot Technology
 *  Copyright (c) 2017 Hesai Photonics Technology, Yang Sheng
 *  Copyright (c) 2020 Hesai Photonics Technology, Lingwen Fang
 *
 *  License: Modified BSD Software License Agreement
 *
 *  $Id: data_base.h 1554 2011-06-14 22:11:17Z jack.oquin $
 */

/** \file
 *
 *  Point Cloud Library point structures for hesai data.
 *
 *  @author Jesse Vera
 *  @author Jack O'Quin
 *  @author Piyush Khandelwal
 *  @author Yang Sheng
 */

// struct PointXYZIT {
//     double x;
//     double y;
//     double z;
//     float intensity;
//     double timestamp;
//     uint16_t ring;                      ///< laser ring number
// } ;

// struct PointCloudHeader {
//   double stamp;
//   uint32_t seq;
//   std::string frame_id;
// };

// struct PointCloud {
//   PointCloudHeader header;
//   std::vector<PointXYZIT> points;
//   uint32_t width;
//   uint32_t height;
//   bool is_dense;

//   inline void resize(int size) {
//     points.resize(size);
//     height = 1;
//     width = size;
//   };

//   inline void clear() {
//     points.clear();
//     height = 0;
//     width = 0;
//   };

//   PointCloud (uint32_t Width, uint32_t Height) {
//     height = Height;
//     width = Width;
//     points.resize(Width * Height);
//   };
// };

