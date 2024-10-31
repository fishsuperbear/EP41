{
  "ShortName": "/HuaweiMDC/FunctionalSoftware/EgoTrajectoryMsgDataType/EgoTrajectory",
  "Namespace": "::ara::egotrajectory",
  "Type": "STRUCTURE",
  "Subelements": [
    {
      "Name": "header",
      "Type": "/HuaweiMDC/FunctionalSoftware/EgoTrajectoryMsgDataType/Header"
    },
    {
      "Name": "locationState",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "pose",
      "Type": "/HuaweiMDC/ApplicationDataType/PoseWithCovariance"
    },
    {
      "Name": "velocity",
      "Type": "/HuaweiMDC/ApplicationDataType/TwistWithCovariance"
    },
    {
      "Name": "acceleration",
      "Type": "/HuaweiMDC/ApplicationDataType/AccelWithCovariance"
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
      "Type": "/HuaweiMDC/FunctionalSoftware/EgoTrajectoryMsgDataType/TrajectoryPointVector"
    },
    {
      "Name": "wayPoints",
      "Type": "/HuaweiMDC/FunctionalSoftware/EgoTrajectoryMsgDataType/WayPointVector"
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
      "Name": "routingHeader",
      "Type": "/HuaweiMDC/FunctionalSoftware/EgoTrajectoryMsgDataType/Header"
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
      "Type": "/HuaweiMDC/FunctionalSoftware/EgoTrajectoryMsgDataType/Estop"
    },
    {
      "Name": "turnLight",
      "Type": "/HuaweiMDC/StdCppBasicDataType/uint8_t"
    },
    {
      "Name": "isHold",
      "Type": "/HuaweiMDC/StdCppBasicDataType/bool"
    }
  ],
  "IsPlane": "false"
}