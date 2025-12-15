// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from ad_msgs:msg/PolyfitLaneDataArray.idl
// generated code does not contain a copyright notice

#ifndef AD_MSGS__MSG__DETAIL__POLYFIT_LANE_DATA_ARRAY__STRUCT_H_
#define AD_MSGS__MSG__DETAIL__POLYFIT_LANE_DATA_ARRAY__STRUCT_H_

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
#include "rosidl_runtime_c/string.h"
// Member 'polyfitlanes'
#include "ad_msgs/msg/detail/polyfit_lane_data__struct.h"

/// Struct defined in msg/PolyfitLaneDataArray in the package ad_msgs.
typedef struct ad_msgs__msg__PolyfitLaneDataArray
{
  rosidl_runtime_c__String frame_id;
  ad_msgs__msg__PolyfitLaneData__Sequence polyfitlanes;
} ad_msgs__msg__PolyfitLaneDataArray;

// Struct for a sequence of ad_msgs__msg__PolyfitLaneDataArray.
typedef struct ad_msgs__msg__PolyfitLaneDataArray__Sequence
{
  ad_msgs__msg__PolyfitLaneDataArray * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} ad_msgs__msg__PolyfitLaneDataArray__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // AD_MSGS__MSG__DETAIL__POLYFIT_LANE_DATA_ARRAY__STRUCT_H_
