// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from ad_msgs:msg/MissionDisplay.idl
// generated code does not contain a copyright notice

#ifndef AD_MSGS__MSG__DETAIL__MISSION_DISPLAY__BUILDER_HPP_
#define AD_MSGS__MSG__DETAIL__MISSION_DISPLAY__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "ad_msgs/msg/detail/mission_display__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace ad_msgs
{

namespace msg
{

namespace builder
{

class Init_MissionDisplay_regions
{
public:
  explicit Init_MissionDisplay_regions(::ad_msgs::msg::MissionDisplay & msg)
  : msg_(msg)
  {}
  ::ad_msgs::msg::MissionDisplay regions(::ad_msgs::msg::MissionDisplay::_regions_type arg)
  {
    msg_.regions = std::move(arg);
    return std::move(msg_);
  }

private:
  ::ad_msgs::msg::MissionDisplay msg_;
};

class Init_MissionDisplay_objects
{
public:
  Init_MissionDisplay_objects()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_MissionDisplay_regions objects(::ad_msgs::msg::MissionDisplay::_objects_type arg)
  {
    msg_.objects = std::move(arg);
    return Init_MissionDisplay_regions(msg_);
  }

private:
  ::ad_msgs::msg::MissionDisplay msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::ad_msgs::msg::MissionDisplay>()
{
  return ad_msgs::msg::builder::Init_MissionDisplay_objects();
}

}  // namespace ad_msgs

#endif  // AD_MSGS__MSG__DETAIL__MISSION_DISPLAY__BUILDER_HPP_
