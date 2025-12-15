// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from ad_msgs:msg/PolyfitLaneDataArray.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "ad_msgs/msg/detail/polyfit_lane_data_array__struct.hpp"
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

void PolyfitLaneDataArray_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) ad_msgs::msg::PolyfitLaneDataArray(_init);
}

void PolyfitLaneDataArray_fini_function(void * message_memory)
{
  auto typed_message = static_cast<ad_msgs::msg::PolyfitLaneDataArray *>(message_memory);
  typed_message->~PolyfitLaneDataArray();
}

size_t size_function__PolyfitLaneDataArray__polyfitlanes(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<ad_msgs::msg::PolyfitLaneData> *>(untyped_member);
  return member->size();
}

const void * get_const_function__PolyfitLaneDataArray__polyfitlanes(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<ad_msgs::msg::PolyfitLaneData> *>(untyped_member);
  return &member[index];
}

void * get_function__PolyfitLaneDataArray__polyfitlanes(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<ad_msgs::msg::PolyfitLaneData> *>(untyped_member);
  return &member[index];
}

void fetch_function__PolyfitLaneDataArray__polyfitlanes(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const ad_msgs::msg::PolyfitLaneData *>(
    get_const_function__PolyfitLaneDataArray__polyfitlanes(untyped_member, index));
  auto & value = *reinterpret_cast<ad_msgs::msg::PolyfitLaneData *>(untyped_value);
  value = item;
}

void assign_function__PolyfitLaneDataArray__polyfitlanes(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<ad_msgs::msg::PolyfitLaneData *>(
    get_function__PolyfitLaneDataArray__polyfitlanes(untyped_member, index));
  const auto & value = *reinterpret_cast<const ad_msgs::msg::PolyfitLaneData *>(untyped_value);
  item = value;
}

void resize_function__PolyfitLaneDataArray__polyfitlanes(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<ad_msgs::msg::PolyfitLaneData> *>(untyped_member);
  member->resize(size);
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember PolyfitLaneDataArray_message_member_array[2] = {
  {
    "frame_id",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ad_msgs::msg::PolyfitLaneDataArray, frame_id),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "polyfitlanes",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<ad_msgs::msg::PolyfitLaneData>(),  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ad_msgs::msg::PolyfitLaneDataArray, polyfitlanes),  // bytes offset in struct
    nullptr,  // default value
    size_function__PolyfitLaneDataArray__polyfitlanes,  // size() function pointer
    get_const_function__PolyfitLaneDataArray__polyfitlanes,  // get_const(index) function pointer
    get_function__PolyfitLaneDataArray__polyfitlanes,  // get(index) function pointer
    fetch_function__PolyfitLaneDataArray__polyfitlanes,  // fetch(index, &value) function pointer
    assign_function__PolyfitLaneDataArray__polyfitlanes,  // assign(index, value) function pointer
    resize_function__PolyfitLaneDataArray__polyfitlanes  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers PolyfitLaneDataArray_message_members = {
  "ad_msgs::msg",  // message namespace
  "PolyfitLaneDataArray",  // message name
  2,  // number of fields
  sizeof(ad_msgs::msg::PolyfitLaneDataArray),
  PolyfitLaneDataArray_message_member_array,  // message members
  PolyfitLaneDataArray_init_function,  // function to initialize message memory (memory has to be allocated)
  PolyfitLaneDataArray_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t PolyfitLaneDataArray_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &PolyfitLaneDataArray_message_members,
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
get_message_type_support_handle<ad_msgs::msg::PolyfitLaneDataArray>()
{
  return &::ad_msgs::msg::rosidl_typesupport_introspection_cpp::PolyfitLaneDataArray_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, ad_msgs, msg, PolyfitLaneDataArray)() {
  return &::ad_msgs::msg::rosidl_typesupport_introspection_cpp::PolyfitLaneDataArray_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif
