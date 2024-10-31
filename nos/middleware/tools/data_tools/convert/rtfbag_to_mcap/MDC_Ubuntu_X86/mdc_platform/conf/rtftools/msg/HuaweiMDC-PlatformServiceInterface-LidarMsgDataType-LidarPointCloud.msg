{
  "ShortName": "/HuaweiMDC/PlatformServiceInterface/LidarMsgDataType/LidarPointCloud",
  "Namespace": "::ara::lidar",
  "Type": "STRUCTURE",
  "Subelements": [
    {
      "Name": "header",
      "Type": "/HuaweiMDC/PlatformServiceInterface/LidarMsgDataType/Header"
    },
    {
      "Name": "isBigEndian",
      "Type": "/HuaweiMDC/StdCppDataType/UInt32"
    },
    {
      "Name": "height",
      "Type": "/HuaweiMDC/StdCppDataType/UInt32"
    },
    {
      "Name": "width",
      "Type": "/HuaweiMDC/StdCppDataType/UInt32"
    },
    {
      "Name": "pointStep",
      "Type": "/HuaweiMDC/StdCppDataType/UInt32"
    },
    {
      "Name": "rowStep",
      "Type": "/HuaweiMDC/StdCppDataType/UInt32"
    },
    {
      "Name": "isDense",
      "Type": "/HuaweiMDC/StdCppDataType/UInt32"
    },
    {
      "Name": "data",
      "Type": "/HuaweiMDC/PlatformServiceInterface/LidarMsgDataType/LidarPointField"
    }
  ],
  "IsPlane": "false"
}