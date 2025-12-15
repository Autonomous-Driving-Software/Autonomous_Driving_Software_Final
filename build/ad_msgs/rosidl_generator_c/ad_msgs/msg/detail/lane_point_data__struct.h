// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from ad_msgs:msg/LanePointData.idl
// generated code does not contain a copyright notice

#ifndef AD_MSGS__MSG__DETAIL__LANE_POINT_DATA__STRUCT_H_
#define AD_MSGS__MSG__DETAIL__LANE_POINT_DATA__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'frame_id'
// Member 'id'
#include "rosidl_runtime_c/string.h"
// Member 'point'
#include "geometry_msgs/msg/detail/point__struct.h"

/// Struct defined in msg/LanePointData in the package ad_msgs.
typedef struct ad_msgs__msg__LanePointData
{
  rosidl_runtime_c__String frame_id;
  rosidl_runtime_c__String id;
  geometry_msgs__msg__Point__Sequence point;
} ad_msgs__msg__LanePointData;

// Struct for a sequence of ad_msgs__msg__LanePointData.
typedef struct ad_msgs__msg__LanePointData__Sequence
{
  ad_msgs__msg__LanePointData * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} ad_msgs__msg__LanePointData__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // AD_MSGS__MSG__DETAIL__LANE_POINT_DATA__STRUCT_H_
