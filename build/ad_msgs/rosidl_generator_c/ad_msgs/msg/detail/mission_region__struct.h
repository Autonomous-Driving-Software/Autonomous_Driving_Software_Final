// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from ad_msgs:msg/MissionRegion.idl
// generated code does not contain a copyright notice

#ifndef AD_MSGS__MSG__DETAIL__MISSION_REGION__STRUCT_H_
#define AD_MSGS__MSG__DETAIL__MISSION_REGION__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'mission'
// Member 'sub_type'
#include "rosidl_runtime_c/string.h"

/// Struct defined in msg/MissionRegion in the package ad_msgs.
typedef struct ad_msgs__msg__MissionRegion
{
  double x;
  double y;
  double radius;
  rosidl_runtime_c__String mission;
  rosidl_runtime_c__String sub_type;
} ad_msgs__msg__MissionRegion;

// Struct for a sequence of ad_msgs__msg__MissionRegion.
typedef struct ad_msgs__msg__MissionRegion__Sequence
{
  ad_msgs__msg__MissionRegion * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} ad_msgs__msg__MissionRegion__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // AD_MSGS__MSG__DETAIL__MISSION_REGION__STRUCT_H_
