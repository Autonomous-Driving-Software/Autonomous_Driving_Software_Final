// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from ad_msgs:msg/PolyfitLaneDataArray.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "ad_msgs/msg/detail/polyfit_lane_data_array__rosidl_typesupport_introspection_c.h"
#include "ad_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "ad_msgs/msg/detail/polyfit_lane_data_array__functions.h"
#include "ad_msgs/msg/detail/polyfit_lane_data_array__struct.h"


// Include directives for member types
// Member `frame_id`
#include "rosidl_runtime_c/string_functions.h"
// Member `polyfitlanes`
#include "ad_msgs/msg/polyfit_lane_data.h"
// Member `polyfitlanes`
#include "ad_msgs/msg/detail/polyfit_lane_data__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void ad_msgs__msg__PolyfitLaneDataArray__rosidl_typesupport_introspection_c__PolyfitLaneDataArray_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  ad_msgs__msg__PolyfitLaneDataArray__init(message_memory);
}

void ad_msgs__msg__PolyfitLaneDataArray__rosidl_typesupport_introspection_c__PolyfitLaneDataArray_fini_function(void * message_memory)
{
  ad_msgs__msg__PolyfitLaneDataArray__fini(message_memory);
}

size_t ad_msgs__msg__PolyfitLaneDataArray__rosidl_typesupport_introspection_c__size_function__PolyfitLaneDataArray__polyfitlanes(
  const void * untyped_member)
{
  const ad_msgs__msg__PolyfitLaneData__Sequence * member =
    (const ad_msgs__msg__PolyfitLaneData__Sequence *)(untyped_member);
  return member->size;
}

const void * ad_msgs__msg__PolyfitLaneDataArray__rosidl_typesupport_introspection_c__get_const_function__PolyfitLaneDataArray__polyfitlanes(
  const void * untyped_member, size_t index)
{
  const ad_msgs__msg__PolyfitLaneData__Sequence * member =
    (const ad_msgs__msg__PolyfitLaneData__Sequence *)(untyped_member);
  return &member->data[index];
}

void * ad_msgs__msg__PolyfitLaneDataArray__rosidl_typesupport_introspection_c__get_function__PolyfitLaneDataArray__polyfitlanes(
  void * untyped_member, size_t index)
{
  ad_msgs__msg__PolyfitLaneData__Sequence * member =
    (ad_msgs__msg__PolyfitLaneData__Sequence *)(untyped_member);
  return &member->data[index];
}

void ad_msgs__msg__PolyfitLaneDataArray__rosidl_typesupport_introspection_c__fetch_function__PolyfitLaneDataArray__polyfitlanes(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const ad_msgs__msg__PolyfitLaneData * item =
    ((const ad_msgs__msg__PolyfitLaneData *)
    ad_msgs__msg__PolyfitLaneDataArray__rosidl_typesupport_introspection_c__get_const_function__PolyfitLaneDataArray__polyfitlanes(untyped_member, index));
  ad_msgs__msg__PolyfitLaneData * value =
    (ad_msgs__msg__PolyfitLaneData *)(untyped_value);
  *value = *item;
}

void ad_msgs__msg__PolyfitLaneDataArray__rosidl_typesupport_introspection_c__assign_function__PolyfitLaneDataArray__polyfitlanes(
  void * untyped_member, size_t index, const void * untyped_value)
{
  ad_msgs__msg__PolyfitLaneData * item =
    ((ad_msgs__msg__PolyfitLaneData *)
    ad_msgs__msg__PolyfitLaneDataArray__rosidl_typesupport_introspection_c__get_function__PolyfitLaneDataArray__polyfitlanes(untyped_member, index));
  const ad_msgs__msg__PolyfitLaneData * value =
    (const ad_msgs__msg__PolyfitLaneData *)(untyped_value);
  *item = *value;
}

bool ad_msgs__msg__PolyfitLaneDataArray__rosidl_typesupport_introspection_c__resize_function__PolyfitLaneDataArray__polyfitlanes(
  void * untyped_member, size_t size)
{
  ad_msgs__msg__PolyfitLaneData__Sequence * member =
    (ad_msgs__msg__PolyfitLaneData__Sequence *)(untyped_member);
  ad_msgs__msg__PolyfitLaneData__Sequence__fini(member);
  return ad_msgs__msg__PolyfitLaneData__Sequence__init(member, size);
}

static rosidl_typesupport_introspection_c__MessageMember ad_msgs__msg__PolyfitLaneDataArray__rosidl_typesupport_introspection_c__PolyfitLaneDataArray_message_member_array[2] = {
  {
    "frame_id",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ad_msgs__msg__PolyfitLaneDataArray, frame_id),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "polyfitlanes",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ad_msgs__msg__PolyfitLaneDataArray, polyfitlanes),  // bytes offset in struct
    NULL,  // default value
    ad_msgs__msg__PolyfitLaneDataArray__rosidl_typesupport_introspection_c__size_function__PolyfitLaneDataArray__polyfitlanes,  // size() function pointer
    ad_msgs__msg__PolyfitLaneDataArray__rosidl_typesupport_introspection_c__get_const_function__PolyfitLaneDataArray__polyfitlanes,  // get_const(index) function pointer
    ad_msgs__msg__PolyfitLaneDataArray__rosidl_typesupport_introspection_c__get_function__PolyfitLaneDataArray__polyfitlanes,  // get(index) function pointer
    ad_msgs__msg__PolyfitLaneDataArray__rosidl_typesupport_introspection_c__fetch_function__PolyfitLaneDataArray__polyfitlanes,  // fetch(index, &value) function pointer
    ad_msgs__msg__PolyfitLaneDataArray__rosidl_typesupport_introspection_c__assign_function__PolyfitLaneDataArray__polyfitlanes,  // assign(index, value) function pointer
    ad_msgs__msg__PolyfitLaneDataArray__rosidl_typesupport_introspection_c__resize_function__PolyfitLaneDataArray__polyfitlanes  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers ad_msgs__msg__PolyfitLaneDataArray__rosidl_typesupport_introspection_c__PolyfitLaneDataArray_message_members = {
  "ad_msgs__msg",  // message namespace
  "PolyfitLaneDataArray",  // message name
  2,  // number of fields
  sizeof(ad_msgs__msg__PolyfitLaneDataArray),
  ad_msgs__msg__PolyfitLaneDataArray__rosidl_typesupport_introspection_c__PolyfitLaneDataArray_message_member_array,  // message members
  ad_msgs__msg__PolyfitLaneDataArray__rosidl_typesupport_introspection_c__PolyfitLaneDataArray_init_function,  // function to initialize message memory (memory has to be allocated)
  ad_msgs__msg__PolyfitLaneDataArray__rosidl_typesupport_introspection_c__PolyfitLaneDataArray_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t ad_msgs__msg__PolyfitLaneDataArray__rosidl_typesupport_introspection_c__PolyfitLaneDataArray_message_type_support_handle = {
  0,
  &ad_msgs__msg__PolyfitLaneDataArray__rosidl_typesupport_introspection_c__PolyfitLaneDataArray_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_ad_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, ad_msgs, msg, PolyfitLaneDataArray)() {
  ad_msgs__msg__PolyfitLaneDataArray__rosidl_typesupport_introspection_c__PolyfitLaneDataArray_message_member_array[1].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, ad_msgs, msg, PolyfitLaneData)();
  if (!ad_msgs__msg__PolyfitLaneDataArray__rosidl_typesupport_introspection_c__PolyfitLaneDataArray_message_type_support_handle.typesupport_identifier) {
    ad_msgs__msg__PolyfitLaneDataArray__rosidl_typesupport_introspection_c__PolyfitLaneDataArray_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &ad_msgs__msg__PolyfitLaneDataArray__rosidl_typesupport_introspection_c__PolyfitLaneDataArray_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
