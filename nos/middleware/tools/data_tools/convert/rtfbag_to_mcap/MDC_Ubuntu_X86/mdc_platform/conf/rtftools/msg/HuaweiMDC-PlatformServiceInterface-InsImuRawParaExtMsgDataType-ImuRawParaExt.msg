{
  "ShortName": "/HuaweiMDC/PlatformServiceInterface/InsImuRawParaExtMsgDataType/ImuRawParaExt",
  "Namespace": "::mdc::imuparaext",
  "Type": "STRUCTURE",
  "Subelements": [
    {
      "Name": "header",
      "Type": "/HuaweiMDC/PlatformServiceInterface/GnssInfoDataType/Header"
    },
    {
      "Name": "extrinsic",
      "Type": "/HuaweiMDC/PlatformServiceInterface/GnssInfoDataType/GeometryPoit"
    },
    {
      "Name": "extValid",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "extTow",
      "Type": "/HuaweiMDC/StdCppDataType/UInt32"
    },
    {
      "Name": "extWeek",
      "Type": "/HuaweiMDC/StdCppDataType/UInt16"
    },
    {
      "Name": "extSyncTime",
      "Type": "/HuaweiMDC/StdCppDataType/UInt64"
    },
    {
      "Name": "extSyncStatus",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
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
      "Type": "/HuaweiMDC/StdCppDataType/UInt64"
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