{
  "ShortName": "/HuaweiMDC/PlatformServiceInterface/DataFlowMsgDataType/DataFlowDataType",
  "Namespace": "::ara::dataflow",
  "Type": "STRUCTURE",
  "Subelements": [
    {
      "Name": "header",
      "Type": "/HuaweiMDC/PlatformServiceInterface/DataFlowMsgDataType/Header"
    },
    {
      "Name": "channel_type",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "channel_id",
      "Type": "/HuaweiMDC/StdCppDataType/String"
    },
    {
      "Name": "recv_flow_stats_pps",
      "Type": "/HuaweiMDC/StdCppDataType/UInt32"
    },
    {
      "Name": "recv_flow_stats_bps",
      "Type": "/HuaweiMDC/StdCppDataType/UInt32"
    },
    {
      "Name": "send_flow_stats_pps",
      "Type": "/HuaweiMDC/StdCppDataType/UInt32"
    },
    {
      "Name": "send_flow_stats_bps",
      "Type": "/HuaweiMDC/StdCppDataType/UInt32"
    }
  ],
  "IsPlane": "false"
}