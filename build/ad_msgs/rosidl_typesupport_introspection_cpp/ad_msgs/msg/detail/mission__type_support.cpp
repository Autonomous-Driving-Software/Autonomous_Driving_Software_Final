// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from ad_msgs:msg/Mission.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "ad_msgs/msg/detail/mission__struct.hpp"
#include "rosidl_typesupport_introspection_cpp/field_types.hpp"
#include "rosidl_typesupport_introspection_cpp/identifier.hpp"
#include "rosidl_typesupport_introspection_cpp/message_introspection.hpp"
#include "rosidl_typesupport_introspection_cpp/message_type_support_decl.hpp"
#include "rosidl_typesupport_introspection_cpp/visibility_control.h"

namespace ad_msgs
{

namespace msg
{

namespace rosidl_typesupport_introspection_cpp
{

void Mission_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) ad_msgs::msg::Mission(_init);
}

void Mission_fini_function(void * message_memory)
{
  auto typed_message = static_cast<ad_msgs::msg::Mission *>(message_memory);
  typed_message->~Mission();
}

size_t size_function__Mission__objects(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<ad_msgs::msg::MissionObject> *>(untyped_member);
  return member->size();
}

const void * get_const_function__Mission__objects(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<ad_msgs::msg::MissionObject> *>(untyped_member);
  return &member[index];
}

void * get_function__Mission__objects(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<ad_msgs::msg::MissionObject> *>(untyped_member);
  return &member[index];
}

void fetch_function__Mission__objects(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const ad_msgs::msg::MissionObject *>(
    get_const_function__Mission__objects(untyped_member, index));
  auto & value = *reinterpret_cast<ad_msgs::msg::MissionObject *>(untyped_value);
  value = item;
}

void assign_function__Mission__objects(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<ad_msgs::msg::MissionObject *>(
    get_function__Mission__objects(untyped_member, index));
  const auto & value = *reinterpret_cast<const ad_msgs::msg::MissionObject *>(untyped_value);
  item = value;
}

void resize_function__Mission__objects(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<ad_msgs::msg::MissionObject> *>(untyped_member);
  member->resize(size);
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember Mission_message_member_array[5] = {
  {
    "objects",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<ad_msgs::msg::MissionObject>(),  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ad_msgs::msg::Mission, objects),  // bytes offset in struct
    nullptr,  // default value
    size_function__Mission__objects,  // size() function pointer
    get_const_function__Mission__objects,  // get_const(index) function pointer
    get_function__Mission__objects,  // get(index) function pointer
    fetch_function__Mission__objects,  // fetch(index, &value) function pointer
    assign_function__Mission__objects,  // assign(index, value) function pointer
    resize_function__Mission__objects  // resize(index) function pointer
  },
  {
    "road_condition",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ad_msgs::msg::Mission, road_condition),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "road_slope",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ad_msgs::msg::Mission, road_slope),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "speed_limit",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ad_msgs::msg::Mission, speed_limit),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "parking",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ad_msgs::msg::Mission, parking),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers Mission_message_members = {
  "ad_msgs::msg",  // message namespace
  "Mission",  // message name
  5,  // number of fields
  sizeof(ad_msgs::msg::Mission),
  Mission_message_member_array,  // message members
  Mission_init_function,  // function to initialize message memory (memory has to be allocated)
  Mission_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t Mission_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &Mission_message_members,
  get_message_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_introspection_cpp

}  // namespace msg

}  // namespace ad_msgs


namespace rosidl_typesupport_introspection_cpp
{

template<>
ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<ad_msgs::msg::Mission>()
{
  return &::ad_msgs::msg::rosidl_typesupport_introspection_cpp::Mission_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, ad_msgs, msg, Mission)() {
  return &::ad_msgs::msg::rosidl_typesupport_introspection_cpp::Mission_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif
