// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from ad_msgs:msg/PolyfitLaneData.idl
// generated code does not contain a copyright notice

#ifndef AD_MSGS__MSG__DETAIL__POLYFIT_LANE_DATA__STRUCT_H_
#define AD_MSGS__MSG__DETAIL__POLYFIT_LANE_DATA__STRUCT_H_

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

/// Struct defined in msg/PolyfitLaneData in the package ad_msgs.
typedef struct ad_msgs__msg__PolyfitLaneData
{
  rosidl_runtime_c__String frame_id;
  rosidl_runtime_c__String id;
  double a0;
  double a1;
  double a2;
  double a3;
} ad_msgs__msg__PolyfitLaneData;

// Struct for a sequence of ad_msgs__msg__PolyfitLaneData.
typedef struct ad_msgs__msg__PolyfitLaneData__Sequence
{
  ad_msgs__msg__PolyfitLaneData * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} ad_msgs__msg__PolyfitLaneData__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // AD_MSGS__MSG__DETAIL__POLYFIT_LANE_DATA__STRUCT_H_
