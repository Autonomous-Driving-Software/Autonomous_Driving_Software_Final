// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from ad_msgs:msg/VehicleCommand.idl
// generated code does not contain a copyright notice

#ifndef AD_MSGS__MSG__DETAIL__VEHICLE_COMMAND__STRUCT_H_
#define AD_MSGS__MSG__DETAIL__VEHICLE_COMMAND__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in msg/VehicleCommand in the package ad_msgs.
typedef struct ad_msgs__msg__VehicleCommand
{
  double steering;
  double accel;
  double brake;
} ad_msgs__msg__VehicleCommand;

// Struct for a sequence of ad_msgs__msg__VehicleCommand.
typedef struct ad_msgs__msg__VehicleCommand__Sequence
{
  ad_msgs__msg__VehicleCommand * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} ad_msgs__msg__VehicleCommand__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // AD_MSGS__MSG__DETAIL__VEHICLE_COMMAND__STRUCT_H_
