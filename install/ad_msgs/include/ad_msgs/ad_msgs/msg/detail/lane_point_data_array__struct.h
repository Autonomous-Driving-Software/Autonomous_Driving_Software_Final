// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from ad_msgs:msg/LanePointDataArray.idl
// generated code does not contain a copyright notice

#ifndef AD_MSGS__MSG__DETAIL__LANE_POINT_DATA_ARRAY__STRUCT_H_
#define AD_MSGS__MSG__DETAIL__LANE_POINT_DATA_ARRAY__STRUCT_H_

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
// Member 'lane'
#include "ad_msgs/msg/detail/lane_point_data__struct.h"

/// Struct defined in msg/LanePointDataArray in the package ad_msgs.
typedef struct ad_msgs__msg__LanePointDataArray
{
  rosidl_runtime_c__String frame_id;
  rosidl_runtime_c__String id;
  ad_msgs__msg__LanePointData__Sequence lane;
} ad_msgs__msg__LanePointDataArray;

// Struct for a sequence of ad_msgs__msg__LanePointDataArray.
typedef struct ad_msgs__msg__LanePointDataArray__Sequence
{
  ad_msgs__msg__LanePointDataArray * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} ad_msgs__msg__LanePointDataArray__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // AD_MSGS__MSG__DETAIL__LANE_POINT_DATA_ARRAY__STRUCT_H_
