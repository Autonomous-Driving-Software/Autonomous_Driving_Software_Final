// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from ad_msgs:msg/PolyfitLaneData.idl
// generated code does not contain a copyright notice

#ifndef AD_MSGS__MSG__DETAIL__POLYFIT_LANE_DATA__TRAITS_HPP_
#define AD_MSGS__MSG__DETAIL__POLYFIT_LANE_DATA__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "ad_msgs/msg/detail/polyfit_lane_data__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace ad_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const PolyfitLaneData & msg,
  std::ostream & out)
{
  out << "{";
  // member: frame_id
  {
    out << "frame_id: ";
    rosidl_generator_traits::value_to_yaml(msg.frame_id, out);
    out << ", ";
  }

  // member: id
  {
    out << "id: ";
    rosidl_generator_traits::value_to_yaml(msg.id, out);
    out << ", ";
  }

  // member: a0
  {
    out << "a0: ";
    rosidl_generator_traits::value_to_yaml(msg.a0, out);
    out << ", ";
  }

  // member: a1
  {
    out << "a1: ";
    rosidl_generator_traits::value_to_yaml(msg.a1, out);
    out << ", ";
  }

  // member: a2
  {
    out << "a2: ";
    rosidl_generator_traits::value_to_yaml(msg.a2, out);
    out << ", ";
  }

  // member: a3
  {
    out << "a3: ";
    rosidl_generator_traits::value_to_yaml(msg.a3, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const PolyfitLaneData & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: frame_id
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "frame_id: ";
    rosidl_generator_traits::value_to_yaml(msg.frame_id, out);
    out << "\n";
  }

  // member: id
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "id: ";
    rosidl_generator_traits::value_to_yaml(msg.id, out);
    out << "\n";
  }

  // member: a0
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "a0: ";
    rosidl_generator_traits::value_to_yaml(msg.a0, out);
    out << "\n";
  }

  // member: a1
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "a1: ";
    rosidl_generator_traits::value_to_yaml(msg.a1, out);
    out << "\n";
  }

  // member: a2
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "a2: ";
    rosidl_generator_traits::value_to_yaml(msg.a2, out);
    out << "\n";
  }

  // member: a3
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "a3: ";
    rosidl_generator_traits::value_to_yaml(msg.a3, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const PolyfitLaneData & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace msg

}  // namespace ad_msgs

namespace rosidl_generator_traits
{

[[deprecated("use ad_msgs::msg::to_block_style_yaml() instead")]]
inline void to_yaml(
  const ad_msgs::msg::PolyfitLaneData & msg,
  std::ostream & out, size_t indentation = 0)
{
  ad_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use ad_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const ad_msgs::msg::PolyfitLaneData & msg)
{
  return ad_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<ad_msgs::msg::PolyfitLaneData>()
{
  return "ad_msgs::msg::PolyfitLaneData";
}

template<>
inline const char * name<ad_msgs::msg::PolyfitLaneData>()
{
  return "ad_msgs/msg/PolyfitLaneData";
}

template<>
struct has_fixed_size<ad_msgs::msg::PolyfitLaneData>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<ad_msgs::msg::PolyfitLaneData>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<ad_msgs::msg::PolyfitLaneData>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // AD_MSGS__MSG__DETAIL__POLYFIT_LANE_DATA__TRAITS_HPP_
