// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from ad_msgs:msg/MissionDisplay.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "ad_msgs/msg/detail/mission_display__rosidl_typesupport_introspection_c.h"
#include "ad_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "ad_msgs/msg/detail/mission_display__functions.h"
#include "ad_msgs/msg/detail/mission_display__struct.h"


// Include directives for member types
// Member `objects`
#include "ad_msgs/msg/mission_object.h"
// Member `objects`
#include "ad_msgs/msg/detail/mission_object__rosidl_typesupport_introspection_c.h"
// Member `regions`
#include "ad_msgs/msg/mission_region.h"
// Member `regions`
#include "ad_msgs/msg/detail/mission_region__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void ad_msgs__msg__MissionDisplay__rosidl_typesupport_introspection_c__MissionDisplay_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  ad_msgs__msg__MissionDisplay__init(message_memory);
}

void ad_msgs__msg__MissionDisplay__rosidl_typesupport_introspection_c__MissionDisplay_fini_function(void * message_memory)
{
  ad_msgs__msg__MissionDisplay__fini(message_memory);
}

size_t ad_msgs__msg__MissionDisplay__rosidl_typesupport_introspection_c__size_function__MissionDisplay__objects(
  const void * untyped_member)
{
  const ad_msgs__msg__MissionObject__Sequence * member =
    (const ad_msgs__msg__MissionObject__Sequence *)(untyped_member);
  return member->size;
}

const void * ad_msgs__msg__MissionDisplay__rosidl_typesupport_introspection_c__get_const_function__MissionDisplay__objects(
  const void * untyped_member, size_t index)
{
  const ad_msgs__msg__MissionObject__Sequence * member =
    (const ad_msgs__msg__MissionObject__Sequence *)(untyped_member);
  return &member->data[index];
}

void * ad_msgs__msg__MissionDisplay__rosidl_typesupport_introspection_c__get_function__MissionDisplay__objects(
  void * untyped_member, size_t index)
{
  ad_msgs__msg__MissionObject__Sequence * member =
    (ad_msgs__msg__MissionObject__Sequence *)(untyped_member);
  return &member->data[index];
}

void ad_msgs__msg__MissionDisplay__rosidl_typesupport_introspection_c__fetch_function__MissionDisplay__objects(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const ad_msgs__msg__MissionObject * item =
    ((const ad_msgs__msg__MissionObject *)
    ad_msgs__msg__MissionDisplay__rosidl_typesupport_introspection_c__get_const_function__MissionDisplay__objects(untyped_member, index));
  ad_msgs__msg__MissionObject * value =
    (ad_msgs__msg__MissionObject *)(untyped_value);
  *value = *item;
}

void ad_msgs__msg__MissionDisplay__rosidl_typesupport_introspection_c__assign_function__MissionDisplay__objects(
  void * untyped_member, size_t index, const void * untyped_value)
{
  ad_msgs__msg__MissionObject * item =
    ((ad_msgs__msg__MissionObject *)
    ad_msgs__msg__MissionDisplay__rosidl_typesupport_introspection_c__get_function__MissionDisplay__objects(untyped_member, index));
  const ad_msgs__msg__MissionObject * value =
    (const ad_msgs__msg__MissionObject *)(untyped_value);
  *item = *value;
}

bool ad_msgs__msg__MissionDisplay__rosidl_typesupport_introspection_c__resize_function__MissionDisplay__objects(
  void * untyped_member, size_t size)
{
  ad_msgs__msg__MissionObject__Sequence * member =
    (ad_msgs__msg__MissionObject__Sequence *)(untyped_member);
  ad_msgs__msg__MissionObject__Sequence__fini(member);
  return ad_msgs__msg__MissionObject__Sequence__init(member, size);
}

size_t ad_msgs__msg__MissionDisplay__rosidl_typesupport_introspection_c__size_function__MissionDisplay__regions(
  const void * untyped_member)
{
  const ad_msgs__msg__MissionRegion__Sequence * member =
    (const ad_msgs__msg__MissionRegion__Sequence *)(untyped_member);
  return member->size;
}

const void * ad_msgs__msg__MissionDisplay__rosidl_typesupport_introspection_c__get_const_function__MissionDisplay__regions(
  const void * untyped_member, size_t index)
{
  const ad_msgs__msg__MissionRegion__Sequence * member =
    (const ad_msgs__msg__MissionRegion__Sequence *)(untyped_member);
  return &member->data[index];
}

void * ad_msgs__msg__MissionDisplay__rosidl_typesupport_introspection_c__get_function__MissionDisplay__regions(
  void * untyped_member, size_t index)
{
  ad_msgs__msg__MissionRegion__Sequence * member =
    (ad_msgs__msg__MissionRegion__Sequence *)(untyped_member);
  return &member->data[index];
}

void ad_msgs__msg__MissionDisplay__rosidl_typesupport_introspection_c__fetch_function__MissionDisplay__regions(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const ad_msgs__msg__MissionRegion * item =
    ((const ad_msgs__msg__MissionRegion *)
    ad_msgs__msg__MissionDisplay__rosidl_typesupport_introspection_c__get_const_function__MissionDisplay__regions(untyped_member, index));
  ad_msgs__msg__MissionRegion * value =
    (ad_msgs__msg__MissionRegion *)(untyped_value);
  *value = *item;
}

void ad_msgs__msg__MissionDisplay__rosidl_typesupport_introspection_c__assign_function__MissionDisplay__regions(
  void * untyped_member, size_t index, const void * untyped_value)
{
  ad_msgs__msg__MissionRegion * item =
    ((ad_msgs__msg__MissionRegion *)
    ad_msgs__msg__MissionDisplay__rosidl_typesupport_introspection_c__get_function__MissionDisplay__regions(untyped_member, index));
  const ad_msgs__msg__MissionRegion * value =
    (const ad_msgs__msg__MissionRegion *)(untyped_value);
  *item = *value;
}

bool ad_msgs__msg__MissionDisplay__rosidl_typesupport_introspection_c__resize_function__MissionDisplay__regions(
  void * untyped_member, size_t size)
{
  ad_msgs__msg__MissionRegion__Sequence * member =
    (ad_msgs__msg__MissionRegion__Sequence *)(untyped_member);
  ad_msgs__msg__MissionRegion__Sequence__fini(member);
  return ad_msgs__msg__MissionRegion__Sequence__init(member, size);
}

static rosidl_typesupport_introspection_c__MessageMember ad_msgs__msg__MissionDisplay__rosidl_typesupport_introspection_c__MissionDisplay_message_member_array[2] = {
  {
    "objects",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ad_msgs__msg__MissionDisplay, objects),  // bytes offset in struct
    NULL,  // default value
    ad_msgs__msg__MissionDisplay__rosidl_typesupport_introspection_c__size_function__MissionDisplay__objects,  // size() function pointer
    ad_msgs__msg__MissionDisplay__rosidl_typesupport_introspection_c__get_const_function__MissionDisplay__objects,  // get_const(index) function pointer
    ad_msgs__msg__MissionDisplay__rosidl_typesupport_introspection_c__get_function__MissionDisplay__objects,  // get(index) function pointer
    ad_msgs__msg__MissionDisplay__rosidl_typesupport_introspection_c__fetch_function__MissionDisplay__objects,  // fetch(index, &value) function pointer
    ad_msgs__msg__MissionDisplay__rosidl_typesupport_introspection_c__assign_function__MissionDisplay__objects,  // assign(index, value) function pointer
    ad_msgs__msg__MissionDisplay__rosidl_typesupport_introspection_c__resize_function__MissionDisplay__objects  // resize(index) function pointer
  },
  {
    "regions",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ad_msgs__msg__MissionDisplay, regions),  // bytes offset in struct
    NULL,  // default value
    ad_msgs__msg__MissionDisplay__rosidl_typesupport_introspection_c__size_function__MissionDisplay__regions,  // size() function pointer
    ad_msgs__msg__MissionDisplay__rosidl_typesupport_introspection_c__get_const_function__MissionDisplay__regions,  // get_const(index) function pointer
    ad_msgs__msg__MissionDisplay__rosidl_typesupport_introspection_c__get_function__MissionDisplay__regions,  // get(index) function pointer
    ad_msgs__msg__MissionDisplay__rosidl_typesupport_introspection_c__fetch_function__MissionDisplay__regions,  // fetch(index, &value) function pointer
    ad_msgs__msg__MissionDisplay__rosidl_typesupport_introspection_c__assign_function__MissionDisplay__regions,  // assign(index, value) function pointer
    ad_msgs__msg__MissionDisplay__rosidl_typesupport_introspection_c__resize_function__MissionDisplay__regions  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers ad_msgs__msg__MissionDisplay__rosidl_typesupport_introspection_c__MissionDisplay_message_members = {
  "ad_msgs__msg",  // message namespace
  "MissionDisplay",  // message name
  2,  // number of fields
  sizeof(ad_msgs__msg__MissionDisplay),
  ad_msgs__msg__MissionDisplay__rosidl_typesupport_introspection_c__MissionDisplay_message_member_array,  // message members
  ad_msgs__msg__MissionDisplay__rosidl_typesupport_introspection_c__MissionDisplay_init_function,  // function to initialize message memory (memory has to be allocated)
  ad_msgs__msg__MissionDisplay__rosidl_typesupport_introspection_c__MissionDisplay_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t ad_msgs__msg__MissionDisplay__rosidl_typesupport_introspection_c__MissionDisplay_message_type_support_handle = {
  0,
  &ad_msgs__msg__MissionDisplay__rosidl_typesupport_introspection_c__MissionDisplay_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_ad_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, ad_msgs, msg, MissionDisplay)() {
  ad_msgs__msg__MissionDisplay__rosidl_typesupport_introspection_c__MissionDisplay_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, ad_msgs, msg, MissionObject)();
  ad_msgs__msg__MissionDisplay__rosidl_typesupport_introspection_c__MissionDisplay_message_member_array[1].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, ad_msgs, msg, MissionRegion)();
  if (!ad_msgs__msg__MissionDisplay__rosidl_typesupport_introspection_c__MissionDisplay_message_type_support_handle.typesupport_identifier) {
    ad_msgs__msg__MissionDisplay__rosidl_typesupport_introspection_c__MissionDisplay_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &ad_msgs__msg__MissionDisplay__rosidl_typesupport_introspection_c__MissionDisplay_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
