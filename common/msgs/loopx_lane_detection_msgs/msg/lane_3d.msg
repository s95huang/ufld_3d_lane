# LaneLine

int8 lane_type # this will be merged with segmentation part in the post processing stage
int8 pos_type # this is defined in the post processing stage

lane_2d lane_2d_data

# 3d lane line data
# LaneLineCubicCurve in car coordinate system
float32 x_start_car_coord
float32 x_end_car_coord
float32 a_car_coord
float32 b_car_coord
float32 c_car_coord
float32 d_car_coord

# LaneLineCubicCurve in camera coordinate system
float32 x_start_cam_coord
float32 x_end_cam_coord
float32 a_cam_coord
float32 b_cam_coord
float32 c_cam_coord
float32 d_cam_coord

# curve camera point set
float32[] x_cam # x in camera coordinate system，real world coordinate , unit: meter
float32[] y_cam # y in camera coordinate system，real world coordinate , unit: meter
float32[] z_cam # z in camera coordinate system，real world coordinate , unit: meter

# curve car coord point set, only on XY plane
float32[] x_car # x in car coordinate system，real world coordinate , unit: meter
float32[] y_car # y in car coordinate system，real world coordinate , unit: meter


# LaneLineType
# uint8 WHITE_DASHED = 0
# uint8 WHITE_SOLID = 1
# uint8 YELLOW_DASHED = 2
# uint8 YELLOW_SOLID = 3

# LaneLinePositionType
# int8 CURB_LEFT = -5
# int8 FOURTH_LEFT = -4
# int8 THIRD_LEFT = -3
# int8 ADJACENT_LEFT = -2
# int8 EGO_LEFT = -1
# int8 EGO_CENTER = 0
# int8 EGO_RIGHT = 1
# int8 ADJACENT_RIGHT = 2
# int8 THIRD_RIGHT = 3
# int8 FOURTH_RIGHT = 4
# int8 CURB_RIGHT = 5
# int8 OTHER = 6
# int8 UNKNOWN = 7




