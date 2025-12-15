// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from ad_msgs:msg/MissionRegion.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "ad_msgs/msg/detail/mission_region__rosidl_typesupport_introspection_c.h"
#include "ad_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "ad_msgs/msg/detail/mission_region__functions.h"
#include "ad_msgs/msg/detail/mission_region__struct.h"


// Include directives for member types
// Member `mission`
// Member `sub_type`
#include "rosidl_runtime_c/string_functions.h"

#ifdef __cplusplus
extern "C"
{
#endif

void ad_msgs__msg__MissionRegion__rosidl_typesupport_introspection_c__MissionRegion_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  ad_msgs__msg__MissionRegion__init(message_memory);
}

void ad_msgs__msg__MissionRegion__rosidl_typesupport_introspection_c__MissionRegion_fini_function(void * message_memory)
{
  ad_msgs__msg__MissionRegion__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember ad_msgs__msg__MissionRegion__rosidl_typesupport_introspection_c__MissionRegion_message_member_array[5] = {
  {
    "x",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ad_msgs__msg__MissionRegion, x),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "y",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ad_msgs__msg__MissionRegion, y),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "radius",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ad_msgs__msg__MissionRegion, radius),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "mission",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ad_msgs__msg__MissionRegion, mission),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "sub_type",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ad_msgs__msg__MissionRegion, sub_type),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers ad_msgs__msg__MissionRegion__rosidl_typesupport_introspection_c__MissionRegion_message_members = {
  "ad_msgs__msg",  // message namespace
  "MissionRegion",  // message name
  5,  // number of fields
  sizeof(ad_msgs__msg__MissionRegion),
  ad_msgs__msg__MissionRegion__rosidl_typesupport_introspection_c__MissionRegion_message_member_array,  // message members
  ad_msgs__msg__MissionRegion__rosidl_typesupport_introspection_c__MissionRegion_init_function,  // function to initialize message memory (memory has to be allocated)
  ad_msgs__msg__MissionRegion__rosidl_typesupport_introspection_c__MissionRegion_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t ad_msgs__msg__MissionRegion__rosidl_typesupport_introspection_c__MissionRegion_message_type_support_handle = {
  0,
  &ad_msgs__msg__MissionRegion__rosidl_typesupport_introspection_c__MissionRegion_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_ad_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, ad_msgs, msg, MissionRegion)() {
  if (!ad_msgs__msg__MissionRegion__rosidl_typesupport_introspection_c__MissionRegion_message_type_support_handle.typesupport_identifier) {
    ad_msgs__msg__MissionRegion__rosidl_typesupport_introspection_c__MissionRegion_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &ad_msgs__msg__MissionRegion__rosidl_typesupport_introspection_c__MissionRegion_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
