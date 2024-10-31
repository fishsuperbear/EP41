{
  "ShortName": "/HuaweiMDC/PlatformServiceInterface/InsImuRawMsgDataType/ImuRawInfo",
  "Namespace": "::mdc::imu",
  "Type": "STRUCTURE",
  "Subelements": [
    {
      "Name": "header",
      "Type": "/HuaweiMDC/PlatformServiceInterface/GnssInfoDataType/Header"
    },
    {
      "Name": "sensorTime",
      "Type": "/HuaweiMDC/PlatformServiceInterface/GnssInfoDataType/Time"
    },
    {
      "Name": "sensorSyncTime",
      "Type": "/HuaweiMDC/PlatformServiceInterface/GnssInfoDataType/Time"
    },
    {
      "Name": "angularVelocity",
      "Type": "/HuaweiMDC/PlatformServiceInterface/GnssInfoDataType/GeometryPoit"
    },
    {
      "Name": "acceleration",
      "Type": "/HuaweiMDC/PlatformServiceInterface/GnssInfoDataType/GeometryPoit"
    },
    {
      "Name": "imuStatus",
      "Type": "/HuaweiMDC/StdCppDataType/UInt32"
    },
    {
      "Name": "imuCalStatus",
      "Type": "/HuaweiMDC/StdCppDataType/UInt32"
    },
    {
      "Name": "temperature",
      "Type": "/HuaweiMDC/StdCppDataType/Float"
    },
    {
      "Name": "reversed1",
      "Type": "/HuaweiMDC/StdCppDataType/UInt64"
    },
    {
      "Name": "reversed2",
      "Type": "/HuaweiMDC/StdCppDataType/UInt64"
    },
    {
      "Name": "reversed3",
      "Type": "/HuaweiMDC/StdCppDataType/Double"
    },
    {
      "Name": "reversed4",
      "Type": "/HuaweiMDC/StdCppDataType/Double"
    },
    {
      "Name": "reversed5",
      "Type": "/HuaweiMDC/StdCppDataType/Double"
    }
  ],
  "IsPlane": "false"
}