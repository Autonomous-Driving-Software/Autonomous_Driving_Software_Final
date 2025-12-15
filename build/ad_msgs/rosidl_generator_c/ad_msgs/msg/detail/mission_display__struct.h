// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from ad_msgs:msg/MissionDisplay.idl
// generated code does not contain a copyright notice

#ifndef AD_MSGS__MSG__DETAIL__MISSION_DISPLAY__STRUCT_H_
#define AD_MSGS__MSG__DETAIL__MISSION_DISPLAY__STRUCT_H_

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
// Member 'regions'
#include "ad_msgs/msg/detail/mission_region__struct.h"

/// Struct defined in msg/MissionDisplay in the package ad_msgs.
typedef struct ad_msgs__msg__MissionDisplay
{
  ad_msgs__msg__MissionObject__Sequence objects;
  ad_msgs__msg__MissionRegion__Sequence regions;
} ad_msgs__msg__MissionDisplay;

// Struct for a sequence of ad_msgs__msg__MissionDisplay.
typedef struct ad_msgs__msg__MissionDisplay__Sequence
{
  ad_msgs__msg__MissionDisplay * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} ad_msgs__msg__MissionDisplay__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // AD_MSGS__MSG__DETAIL__MISSION_DISPLAY__STRUCT_H_
