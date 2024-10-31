{
  "ShortName": "/DataType/StdCppImplDataType/di/EcuUpdateStatus",
  "Namespace": "::symbol_0",
  "Type": "STRUCTURE",
  "Subelements": [
    {
      "Name": "ecuIdentity",
      "Type": "/DataType/StdCppImplDataType/di/EcuIdentity"
    },
    {
      "Name": "updateType",
      "Type": "/HuaweiMDC/StdCppDataType/UInt32"
    },
    {
      "Name": "state",
      "Type": "/HuaweiMDC/StdCppDataType/UInt32"
    },
    {
      "Name": "retriedTimes",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "serviceErrorCode",
      "Type": "/HuaweiMDC/StdCppDataType/UInt32"
    },
    {
      "Name": "progress",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8",
      "IsOptional": "false"
    },
    {
      "Name": "ecuErrorCode",
      "Type": "/DataType/StdCppImplDataType/di/EcuErrorCode",
      "IsOptional": "false"
    }
  ],
  "IsPlane": "false"
}