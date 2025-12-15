// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from ad_msgs:msg/Mission.idl
// generated code does not contain a copyright notice

#ifndef AD_MSGS__MSG__DETAIL__MISSION__TRAITS_HPP_
#define AD_MSGS__MSG__DETAIL__MISSION__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "ad_msgs/msg/detail/mission__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'objects'
#include "ad_msgs/msg/detail/mission_object__traits.hpp"

namespace ad_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const Mission & msg,
  std::ostream & out)
{
  out << "{";
  // member: objects
  {
    if (msg.objects.size() == 0) {
      out << "objects: []";
    } else {
      out << "objects: [";
      size_t pending_items = msg.objects.size();
      for (auto item : msg.objects) {
        to_flow_style_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: road_condition
  {
    out << "road_condition: ";
    rosidl_generator_traits::value_to_yaml(msg.road_condition, out);
    out << ", ";
  }

  // member: road_slope
  {
    out << "road_slope: ";
    rosidl_generator_traits::value_to_yaml(msg.road_slope, out);
    out << ", ";
  }

  // member: speed_limit
  {
    out << "speed_limit: ";
    rosidl_generator_traits::value_to_yaml(msg.speed_limit, out);
    out << ", ";
  }

  // member: parking
  {
    out << "parking: ";
    rosidl_generator_traits::value_to_yaml(msg.parking, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const Mission & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: objects
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.objects.size() == 0) {
      out << "objects: []\n";
    } else {
      out << "objects:\n";
      for (auto item : msg.objects) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "-\n";
        to_block_style_yaml(item, out, indentation + 2);
      }
    }
  }

  // member: road_condition
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "road_condition: ";
    rosidl_generator_traits::value_to_yaml(msg.road_condition, out);
    out << "\n";
  }

  // member: road_slope
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "road_slope: ";
    rosidl_generator_traits::value_to_yaml(msg.road_slope, out);
    out << "\n";
  }

  // member: speed_limit
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "speed_limit: ";
    rosidl_generator_traits::value_to_yaml(msg.speed_limit, out);
    out << "\n";
  }

  // member: parking
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "parking: ";
    rosidl_generator_traits::value_to_yaml(msg.parking, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const Mission & msg, bool use_flow_style = false)
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
  const ad_msgs::msg::Mission & msg,
  std::ostream & out, size_t indentation = 0)
{
  ad_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use ad_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const ad_msgs::msg::Mission & msg)
{
  return ad_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<ad_msgs::msg::Mission>()
{
  return "ad_msgs::msg::Mission";
}

template<>
inline const char * name<ad_msgs::msg::Mission>()
{
  return "ad_msgs/msg/Mission";
}

template<>
struct has_fixed_size<ad_msgs::msg::Mission>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<ad_msgs::msg::Mission>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<ad_msgs::msg::Mission>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // AD_MSGS__MSG__DETAIL__MISSION__TRAITS_HPP_
