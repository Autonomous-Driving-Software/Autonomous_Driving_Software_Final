// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from ad_msgs:msg/MissionObject.idl
// generated code does not contain a copyright notice

#ifndef AD_MSGS__MSG__DETAIL__MISSION_OBJECT__STRUCT_H_
#define AD_MSGS__MSG__DETAIL__MISSION_OBJECT__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'object_type'
#include "rosidl_runtime_c/string.h"

/// Struct defined in msg/MissionObject in the package ad_msgs.
typedef struct ad_msgs__msg__MissionObject
{
  rosidl_runtime_c__String object_type;
  bool is_reach_end;
  double x;
  double y;
  double yaw;
  double velocity;
} ad_msgs__msg__MissionObject;

// Struct for a sequence of ad_msgs__msg__MissionObject.
typedef struct ad_msgs__msg__MissionObject__Sequence
{
  ad_msgs__msg__MissionObject * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} ad_msgs__msg__MissionObject__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // AD_MSGS__MSG__DETAIL__MISSION_OBJECT__STRUCT_H_
