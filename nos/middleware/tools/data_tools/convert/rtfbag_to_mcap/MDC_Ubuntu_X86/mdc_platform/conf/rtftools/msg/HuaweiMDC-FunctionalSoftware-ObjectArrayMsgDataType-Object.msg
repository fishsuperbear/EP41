{
  "ShortName": "/HuaweiMDC/FunctionalSoftware/ObjectArrayMsgDataType/Object",
  "Namespace": "::ara::adsfi",
  "Type": "STRUCTURE",
  "Subelements": [
    {
      "Name": "header",
      "Type": "/HuaweiMDC/CommonDataTypes/CommonHeader"
    },
    {
      "Name": "existence_probability",
      "Type": "/HuaweiMDC/StdCppDataType/Float"
    },
    {
      "Name": "id",
      "Type": "/HuaweiMDC/StdCppDataType/Int32"
    },
    {
      "Name": "time_last_update",
      "Type": "/HuaweiMDC/CommonDataTypes/CommonHeader"
    },
    {
      "Name": "time_creation",
      "Type": "/HuaweiMDC/CommonDataTypes/CommonHeader"
    },
    {
      "Name": "measurement_age_count",
      "Type": "/HuaweiMDC/StdCppDataType/Int32"
    },
    {
      "Name": "classification",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "classification_description",
      "Type": "/HuaweiMDC/StdCppDataType/String"
    },
    {
      "Name": "classification_confidence",
      "Type": "/HuaweiMDC/StdCppDataType/Float"
    },
    {
      "Name": "classification_age_seconds",
      "Type": "/HuaweiMDC/StdCppDataType/Float"
    },
    {
      "Name": "classification_age_count",
      "Type": "/HuaweiMDC/StdCppDataType/Int32"
    },
    {
      "Name": "object_box_center",
      "Type": "/HuaweiMDC/ApplicationDataType/Point"
    },
    {
      "Name": "object_box_center_absolute",
      "Type": "/HuaweiMDC/ApplicationDataType/Point"
    },
    {
      "Name": "object_box_center_covariance",
      "Type": "/HuaweiMDC/ApplicationDataType/Point"
    },
    {
      "Name": "object_box_size",
      "Type": "/HuaweiMDC/ApplicationDataType/Point"
    },
    {
      "Name": "object_box_size_covariance",
      "Type": "/HuaweiMDC/ApplicationDataType/Point"
    },
    {
      "Name": "box_image",
      "Type": "/HuaweiMDC/ApplicationDataType/Rect"
    },
    {
      "Name": "object_box_orientation",
      "Type": "/HuaweiMDC/StdCppDataType/Float"
    },
    {
      "Name": "object_box_orientation_absolute",
      "Type": "/HuaweiMDC/StdCppDataType/Float"
    },
    {
      "Name": "object_box_orientation_covariance",
      "Type": "/HuaweiMDC/StdCppDataType/Float"
    },
    {
      "Name": "box_polygon",
      "Type": "/HuaweiMDC/ApplicationDataType/Polygon"
    },
    {
      "Name": "box_polygon_absolute",
      "Type": "/HuaweiMDC/ApplicationDataType/PointArray"
    },
    {
      "Name": "reference_point",
      "Type": "/HuaweiMDC/ApplicationDataType/Point"
    },
    {
      "Name": "reference_point_covariance",
      "Type": "/HuaweiMDC/ApplicationDataType/Point"
    },
    {
      "Name": "relative_velocity",
      "Type": "/HuaweiMDC/ApplicationDataType/TwistWithCovariance"
    },
    {
      "Name": "absolute_velocity",
      "Type": "/HuaweiMDC/ApplicationDataType/TwistWithCovariance"
    },
    {
      "Name": "enu_velocity",
      "Type": "/HuaweiMDC/ApplicationDataType/TwistWithCovariance"
    },
    {
      "Name": "sl_velocity",
      "Type": "/HuaweiMDC/ApplicationDataType/PointArray"
    },
    {
      "Name": "contour_points_absolute",
      "Type": "/HuaweiMDC/ApplicationDataType/PointArray"
    },
    {
      "Name": "contour_points",
      "Type": "/HuaweiMDC/ApplicationDataType/Polygon"
    },
    {
      "Name": "intention_paths",
      "Type": "/HuaweiMDC/ApplicationDataType/PathWithCovarianceArray"
    },
    {
      "Name": "intention_time_step",
      "Type": "/HuaweiMDC/StdCppDataType/Float"
    },
    {
      "Name": "lane_id",
      "Type": "/HuaweiMDC/ApplicationDataType/Int32Array"
    },
    {
      "Name": "lane_type",
      "Type": "/HuaweiMDC/ApplicationDataType/Uint8Array"
    },
    {
      "Name": "position_in_lane",
      "Type": "/HuaweiMDC/ApplicationDataType/PointArray"
    },
    {
      "Name": "length_in_lane",
      "Type": "/HuaweiMDC/ApplicationDataType/FloatArray"
    },
    {
      "Name": "width_in_lane",
      "Type": "/HuaweiMDC/ApplicationDataType/FloatArray"
    },
    {
      "Name": "road_id",
      "Type": "/HuaweiMDC/StdCppDataType/Int32"
    },
    {
      "Name": "road_type",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "types",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "cipv_flag",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "fusion_type",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "blinker_flag",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "acceleration",
      "Type": "/HuaweiMDC/ApplicationDataType/Point"
    },
    {
      "Name": "abs_acceleration",
      "Type": "/HuaweiMDC/ApplicationDataType/Point"
    },
    {
      "Name": "anchor_point",
      "Type": "/HuaweiMDC/ApplicationDataType/Point"
    },
    {
      "Name": "abs_anchor_point",
      "Type": "/HuaweiMDC/ApplicationDataType/Point"
    },
    {
      "Name": "acceleration_covariance",
      "Type": "/HuaweiMDC/ApplicationDataType/Matrix3d"
    },
    {
      "Name": "abs_acceleration_covariance",
      "Type": "/HuaweiMDC/ApplicationDataType/Matrix3d"
    },
    {
      "Name": "velocity_covariance",
      "Type": "/HuaweiMDC/ApplicationDataType/Matrix3d"
    },
    {
      "Name": "abs_velocity_covariance",
      "Type": "/HuaweiMDC/ApplicationDataType/Matrix3d"
    },
    {
      "Name": "blinkerStatus",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "cameraStatus",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "coordinate",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    }
  ],
  "IsPlane": "false"
}