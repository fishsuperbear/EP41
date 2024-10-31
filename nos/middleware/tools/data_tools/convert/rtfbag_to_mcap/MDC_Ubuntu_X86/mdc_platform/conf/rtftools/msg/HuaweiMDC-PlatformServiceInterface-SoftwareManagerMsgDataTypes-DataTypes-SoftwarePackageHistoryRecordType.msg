{
  "ShortName": "/HuaweiMDC/PlatformServiceInterface/SoftwareManagerMsgDataTypes/DataTypes/SoftwarePackageHistoryRecordType",
  "Namespace": "::mdc::swm",
  "Type": "STRUCTURE",
  "Subelements": [
    {
      "Name": "id",
      "Type": "/HuaweiMDC/StdCppDataType/Int32"
    },
    {
      "Name": "startTime",
      "Type": "/HuaweiMDC/StdCppDataType/UInt64"
    },
    {
      "Name": "endTime",
      "Type": "/HuaweiMDC/StdCppDataType/UInt64"
    },
    {
      "Name": "softwarePackageName",
      "Type": "/HuaweiMDC/StdCppDataType/String"
    },
    {
      "Name": "version",
      "Type": "/HuaweiMDC/StdCppDataType/String"
    },
    {
      "Name": "errorCode",
      "Type": "/HuaweiMDC/StdCppDataType/Int32"
    },
    {
      "Name": "componentRecord",
      "Type": "/HuaweiMDC/PlatformServiceInterface/SoftwareManagerMsgDataTypes/DataTypes/ComponentHistoryRecordVector"
    },
    {
      "Name": "message",
      "Type": "/HuaweiMDC/StdCppDataType/String"
    }
  ],
  "IsPlane": "false"
}