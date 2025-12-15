// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from ad_msgs:msg/Mission.idl
// generated code does not contain a copyright notice

#ifndef AD_MSGS__MSG__DETAIL__MISSION__STRUCT_H_
#define AD_MSGS__MSG__DETAIL__MISSION__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'objects'
#include "ad_msgs/msg/detail/mission_object__struct.h"
// Member 'road_condition'
// Member 'road_slope'
#include "rosidl_runtime_c/string.h"

/// Struct defined in msg/Mission in the package ad_msgs.
typedef struct ad_msgs__msg__Mission
{
  ad_msgs__msg__MissionObject__Sequence objects;
  rosidl_runtime_c__String road_condition;
  rosidl_runtime_c__String road_slope;
  double speed_limit;
  bool parking;
} ad_msgs__msg__Mission;

// Struct for a sequence of ad_msgs__msg__Mission.
typedef struct ad_msgs__msg__Mission__Sequence
{
  ad_msgs__msg__Mission * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} ad_msgs__msg__Mission__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // AD_MSGS__MSG__DETAIL__MISSION__STRUCT_H_
