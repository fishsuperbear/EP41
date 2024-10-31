{
  "ShortName": "/HuaweiMDC/FunctionalSoftware/PlanningResultMsgDataType/PlanningResult",
  "Namespace": "::adsfi",
  "Type": "STRUCTURE",
  "Subelements": [
    {
      "Name": "header",
      "Type": "/HuaweiMDC/CommonDataTypes/CommonHeader"
    },
    {
      "Name": "trajectoryLength",
      "Type": "/HuaweiMDC/StdCppDataType/Double"
    },
    {
      "Name": "trajectoryPeriod",
      "Type": "/HuaweiMDC/StdCppDataType/Double"
    },
    {
      "Name": "trajectoryPoints",
      "Type": "/HuaweiMDC/FunctionalSoftware/PlanningResultMsgDataType/TrajectoryPointVector"
    },
    {
      "Name": "wayPoints",
      "Type": "/HuaweiMDC/FunctionalSoftware/PlanningResultMsgDataType/WayPointVector"
    },
    {
      "Name": "isReplanning",
      "Type": "/HuaweiMDC/StdCppDataType/Boolean"
    },
    {
      "Name": "gear",
      "Type": "/HuaweiMDC/StdCppDataType/UInt32"
    },
    {
      "Name": "selfLaneId",
      "Type": "/HuaweiMDC/StdCppDataType/String"
    },
    {
      "Name": "trajectoryType",
      "Type": "/HuaweiMDC/StdCppDataType/UInt32"
    },
    {
      "Name": "targetLaneId",
      "Type": "/HuaweiMDC/StdCppDataType/String"
    },
    {
      "Name": "estop",
      "Type": "/HuaweiMDC/FunctionalSoftware/PlanningResultMsgDataType/Estop"
    },
    {
      "Name": "turnLight",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "isHold",
      "Type": "/HuaweiMDC/StdCppDataType/Boolean"
    }
  ],
  "IsPlane": "false"
}