{
  "ShortName": "/HuaweiMDC/FunctionalSoftware/LaneLineArrayMsgDataType/LaneLine",
  "Namespace": "::ara::adsfi",
  "Type": "STRUCTURE",
  "Subelements": [
    {
      "Name": "id",
      "Type": "/HuaweiMDC/StdCppDataType/Int32"
    },
    {
      "Name": "header",
      "Type": "/HuaweiMDC/FunctionalSoftware/LaneLineArrayMsgDataType/LaneHeader"
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
      "Name": "lane_image",
      "Type": "/HuaweiMDC/FunctionalSoftware/LaneLineArrayMsgDataType/LaneLineFitParam"
    },
    {
      "Name": "start_point",
      "Type": "/HuaweiMDC/ApplicationDataType/Pointxy"
    },
    {
      "Name": "end_point",
      "Type": "/HuaweiMDC/ApplicationDataType/Pointxy"
    },
    {
      "Name": "point_list",
      "Type": "/HuaweiMDC/ApplicationDataType/PointxyArray"
    }
  ],
  "IsPlane": "false"
}