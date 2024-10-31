{
  "ShortName": "/HuaweiMDC/PlatformServiceInterface/TsyncDpStatusMsgTypes/DataTypes/TsyncDpStatusType",
  "Namespace": "::ara::tsync::monitor",
  "Type": "STRUCTURE",
  "Subelements": [
    {
      "Name": "header",
      "Type": "/HuaweiMDC/CommonDataTypes/CommonHeader"
    },
    {
      "Name": "centerDpStatus",
      "Type": "/HuaweiMDC/PlatformServiceInterface/TsyncDpStatusMsgTypes/DataTypes/StatusType"
    },
    {
      "Name": "osTimeOffset",
      "Type": "/HuaweiMDC/StdCppDataType/Int64"
    },
    {
      "Name": "portTimeOffsets",
      "Type": "/HuaweiMDC/PlatformServiceInterface/TsyncDpStatusMsgTypes/DataTypes/TimeOffsetsType"
    }
  ],
  "IsPlane": "false"
}