// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from ad_msgs:msg/PolyfitLaneData.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "ad_msgs/msg/detail/polyfit_lane_data__rosidl_typesupport_introspection_c.h"
#include "ad_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "ad_msgs/msg/detail/polyfit_lane_data__functions.h"
#include "ad_msgs/msg/detail/polyfit_lane_data__struct.h"


// Include directives for member types
// Member `frame_id`
// Member `id`
#include "rosidl_runtime_c/string_functions.h"

#ifdef __cplusplus
extern "C"
{
#endif

void ad_msgs__msg__PolyfitLaneData__rosidl_typesupport_introspection_c__PolyfitLaneData_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  ad_msgs__msg__PolyfitLaneData__init(message_memory);
}

void ad_msgs__msg__PolyfitLaneData__rosidl_typesupport_introspection_c__PolyfitLaneData_fini_function(void * message_memory)
{
  ad_msgs__msg__PolyfitLaneData__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember ad_msgs__msg__PolyfitLaneData__rosidl_typesupport_introspection_c__PolyfitLaneData_message_member_array[6] = {
  {
    "frame_id",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ad_msgs__msg__PolyfitLaneData, frame_id),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "id",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ad_msgs__msg__PolyfitLaneData, id),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "a0",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ad_msgs__msg__PolyfitLaneData, a0),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "a1",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ad_msgs__msg__PolyfitLaneData, a1),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "a2",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ad_msgs__msg__PolyfitLaneData, a2),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "a3",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ad_msgs__msg__PolyfitLaneData, a3),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers ad_msgs__msg__PolyfitLaneData__rosidl_typesupport_introspection_c__PolyfitLaneData_message_members = {
  "ad_msgs__msg",  // message namespace
  "PolyfitLaneData",  // message name
  6,  // number of fields
  sizeof(ad_msgs__msg__PolyfitLaneData),
  ad_msgs__msg__PolyfitLaneData__rosidl_typesupport_introspection_c__PolyfitLaneData_message_member_array,  // message members
  ad_msgs__msg__PolyfitLaneData__rosidl_typesupport_introspection_c__PolyfitLaneData_init_function,  // function to initialize message memory (memory has to be allocated)
  ad_msgs__msg__PolyfitLaneData__rosidl_typesupport_introspection_c__PolyfitLaneData_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t ad_msgs__msg__PolyfitLaneData__rosidl_typesupport_introspection_c__PolyfitLaneData_message_type_support_handle = {
  0,
  &ad_msgs__msg__PolyfitLaneData__rosidl_typesupport_introspection_c__PolyfitLaneData_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_ad_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, ad_msgs, msg, PolyfitLaneData)() {
  if (!ad_msgs__msg__PolyfitLaneData__rosidl_typesupport_introspection_c__PolyfitLaneData_message_type_support_handle.typesupport_identifier) {
    ad_msgs__msg__PolyfitLaneData__rosidl_typesupport_introspection_c__PolyfitLaneData_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &ad_msgs__msg__PolyfitLaneData__rosidl_typesupport_introspection_c__PolyfitLaneData_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
