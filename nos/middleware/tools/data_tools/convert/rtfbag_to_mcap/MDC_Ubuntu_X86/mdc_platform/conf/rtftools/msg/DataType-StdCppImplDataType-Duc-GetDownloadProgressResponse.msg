{
  "ShortName": "/DataType/StdCppImplDataType/Duc/GetDownloadProgressResponse",
  "Namespace": "::symbol_0",
  "Type": "STRUCTURE",
  "Subelements": [
    {
      "Name": "responseBase",
      "Type": "/DataType/StdCppImplDataType/Duc/ResponseBase"
    },
    {
      "Name": "state",
      "Type": "/HuaweiMDC/StdCppDataType/UInt32"
    },
    {
      "Name": "progress",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "downloadedBytes",
      "Type": "/HuaweiMDC/StdCppDataType/Int64",
      "IsOptional": "false"
    },
    {
      "Name": "totalBytes",
      "Type": "/HuaweiMDC/StdCppDataType/Int64",
      "IsOptional": "false"
    },
    {
      "Name": "downloadSpeed",
      "Type": "/HuaweiMDC/StdCppDataType/Int64"
    },
    {
      "Name": "ecuDownloadStatuses",
      "Type": "/DataType/StdCppImplDataType/Duc/EcuDownloadStatusList",
      "IsOptional": "false"
    }
  ],
  "IsPlane": "false"
}