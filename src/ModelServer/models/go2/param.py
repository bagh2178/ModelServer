from sensor_msgs.msg import PointField

DT_MAP = {
    PointField.FLOAT32: 'f4',
    PointField.FLOAT64: 'f8',
    PointField.INT32  : 'i4',
    PointField.UINT32 : 'u4',
    PointField.INT16  : 'i2',
    PointField.UINT16 : 'u2',
    PointField.INT8   : 'i1',
    PointField.UINT8  : 'u1',
}

KP = 60.0
KD = 5.0

LegID = {
    "FR_0": 0,  # Front right hip
    "FR_1": 1,  # Front right thigh
    "FR_2": 2,  # Front right calf
    "FL_0": 3,
    "FL_1": 4,
    "FL_2": 5,
    "RR_0": 6,
    "RR_1": 7,
    "RR_2": 8,
    "RL_0": 9,
    "RL_1": 10,
    "RL_2": 11,
}

HIGHLEVEL = 0xEE
LOWLEVEL = 0xFF
TRIGERLEVEL = 0xF0
PosStopF = 2.146e9
VelStopF = 16000.0