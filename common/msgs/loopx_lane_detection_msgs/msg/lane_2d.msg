int32 id # lane id, unique in one frame, -1 means invalid lane, starts from 0
int32 lane_type # lane type
float32 lane_type_per # lane type percentage

float32 confidence # between 0 to 1

int32 image_width
int32 image_height

# lane points, 2d points in image coordinate system (filterd lane points)
float32[] u_points
float32[] v_points

# LaneLineCubicCurve in image coordinate system 
float32 x_start
float32 x_end

# y = ax^3 + bx^2 + cx + d, x is in image u coordinate, y is in image v coordinate
float32 a
float32 b
float32 c
float32 d

# y = k * x + b, x is in image u coordinate, y is in image v coordinate
float32 k
float32 offset