{
  "ShortName": "/HuaweiMDC/MdcPlatformServices/MsgDataType/SomeipMsgDataType/Raw180DataGnss",
  "Namespace": "::sensor::ins180::gnss",
  "Type": "STRUCTURE",
  "Subelements": [
    {
      "Name": "gnssNumSV",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "gnssFlagPos",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "gnssRtkDlyHdm",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8",
      "Padding": "5"
    },
    {
      "Name": "gnssLatitude",
      "Type": "/HuaweiMDC/StdCppDataType/Double"
    },
    {
      "Name": "gnssLongtitude",
      "Type": "/HuaweiMDC/StdCppDataType/Double"
    },
    {
      "Name": "gnssLocatHeight",
      "Type": "/HuaweiMDC/StdCppDataType/Float"
    },
    {
      "Name": "gnssHeading",
      "Type": "/HuaweiMDC/StdCppDataType/Float"
    },
    {
      "Name": "gnssPitch",
      "Type": "/HuaweiMDC/StdCppDataType/Float"
    },
    {
      "Name": "gnssHorizontalSpd",
      "Type": "/HuaweiMDC/StdCppDataType/Float"
    },
    {
      "Name": "gnssVerticalSpd",
      "Type": "/HuaweiMDC/StdCppDataType/Float"
    },
    {
      "Name": "gnssTrackTrue",
      "Type": "/HuaweiMDC/StdCppDataType/Float"
    },
    {
      "Name": "gnssStdLat",
      "Type": "/HuaweiMDC/StdCppDataType/Float"
    },
    {
      "Name": "gnssStdLon",
      "Type": "/HuaweiMDC/StdCppDataType/Float"
    },
    {
      "Name": "gnssStdLocatHeight",
      "Type": "/HuaweiMDC/StdCppDataType/Float"
    },
    {
      "Name": "gnssStdHeading",
      "Type": "/HuaweiMDC/StdCppDataType/Float"
    },
    {
      "Name": "gnssStdPitch",
      "Type": "/HuaweiMDC/StdCppDataType/Float"
    },
    {
      "Name": "gnssStdHorizontalSpd",
      "Type": "/HuaweiMDC/StdCppDataType/Float"
    },
    {
      "Name": "gnssStdVerticalSpd",
      "Type": "/HuaweiMDC/StdCppDataType/Float"
    },
    {
      "Name": "gnssStdTrackTrue",
      "Type": "/HuaweiMDC/StdCppDataType/Float"
    },
    {
      "Name": "gnssGdop",
      "Type": "/HuaweiMDC/StdCppDataType/UInt16"
    },
    {
      "Name": "gnssPdop",
      "Type": "/HuaweiMDC/StdCppDataType/UInt16"
    },
    {
      "Name": "gnssHdop",
      "Type": "/HuaweiMDC/StdCppBasicDataType/uint16_t"
    },
    {
      "Name": "gnssHtdop",
      "Type": "/HuaweiMDC/StdCppDataType/UInt16"
    },
    {
      "Name": "ssrIntegrity",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "gnssValid",
      "Type": "/HuaweiMDC/StdCppDataType/Boolean"
    },
    {
      "Name": "gnssWeek",
      "Type": "/HuaweiMDC/StdCppDataType/UInt16"
    },
    {
      "Name": "gnssTow",
      "Type": "/HuaweiMDC/StdCppDataType/UInt32"
    },
    {
      "Name": "gnssSyncTime",
      "Type": "/HuaweiMDC/StdCppDataType/UInt64"
    },
    {
      "Name": "gnssSyncStatus",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8",
      "Padding": "7"
    }
  ],
  "IsPlane": "true"
}