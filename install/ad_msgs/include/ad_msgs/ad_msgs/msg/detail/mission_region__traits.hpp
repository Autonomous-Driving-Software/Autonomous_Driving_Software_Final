// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from ad_msgs:msg/MissionRegion.idl
// generated code does not contain a copyright notice

#ifndef AD_MSGS__MSG__DETAIL__MISSION_REGION__TRAITS_HPP_
#define AD_MSGS__MSG__DETAIL__MISSION_REGION__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "ad_msgs/msg/detail/mission_region__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace ad_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const MissionRegion & msg,
  std::ostream & out)
{
  out << "{";
  // member: x
  {
    out << "x: ";
    rosidl_generator_traits::value_to_yaml(msg.x, out);
    out << ", ";
  }

  // member: y
  {
    out << "y: ";
    rosidl_generator_traits::value_to_yaml(msg.y, out);
    out << ", ";
  }

  // member: radius
  {
    out << "radius: ";
    rosidl_generator_traits::value_to_yaml(msg.radius, out);
    out << ", ";
  }

  // member: mission
  {
    out << "mission: ";
    rosidl_generator_traits::value_to_yaml(msg.mission, out);
    out << ", ";
  }

  // member: sub_type
  {
    out << "sub_type: ";
    rosidl_generator_traits::value_to_yaml(msg.sub_type, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const MissionRegion & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: x
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "x: ";
    rosidl_generator_traits::value_to_yaml(msg.x, out);
    out << "\n";
  }

  // member: y
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "y: ";
    rosidl_generator_traits::value_to_yaml(msg.y, out);
    out << "\n";
  }

  // member: radius
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "radius: ";
    rosidl_generator_traits::value_to_yaml(msg.radius, out);
    out << "\n";
  }

  // member: mission
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "mission: ";
    rosidl_generator_traits::value_to_yaml(msg.mission, out);
    out << "\n";
  }

  // member: sub_type
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "sub_type: ";
    rosidl_generator_traits::value_to_yaml(msg.sub_type, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const MissionRegion & msg, bool use_flow_style = false)
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
  const ad_msgs::msg::MissionRegion & msg,
  std::ostream & out, size_t indentation = 0)
{
  ad_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use ad_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const ad_msgs::msg::MissionRegion & msg)
{
  return ad_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<ad_msgs::msg::MissionRegion>()
{
  return "ad_msgs::msg::MissionRegion";
}

template<>
inline const char * name<ad_msgs::msg::MissionRegion>()
{
  return "ad_msgs/msg/MissionRegion";
}

template<>
struct has_fixed_size<ad_msgs::msg::MissionRegion>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<ad_msgs::msg::MissionRegion>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<ad_msgs::msg::MissionRegion>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // AD_MSGS__MSG__DETAIL__MISSION_REGION__TRAITS_HPP_
