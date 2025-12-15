// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from ad_msgs:msg/MissionRegion.idl
// generated code does not contain a copyright notice

#ifndef AD_MSGS__MSG__DETAIL__MISSION_REGION__BUILDER_HPP_
#define AD_MSGS__MSG__DETAIL__MISSION_REGION__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "ad_msgs/msg/detail/mission_region__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace ad_msgs
{

namespace msg
{

namespace builder
{

class Init_MissionRegion_sub_type
{
public:
  explicit Init_MissionRegion_sub_type(::ad_msgs::msg::MissionRegion & msg)
  : msg_(msg)
  {}
  ::ad_msgs::msg::MissionRegion sub_type(::ad_msgs::msg::MissionRegion::_sub_type_type arg)
  {
    msg_.sub_type = std::move(arg);
    return std::move(msg_);
  }

private:
  ::ad_msgs::msg::MissionRegion msg_;
};

class Init_MissionRegion_mission
{
public:
  explicit Init_MissionRegion_mission(::ad_msgs::msg::MissionRegion & msg)
  : msg_(msg)
  {}
  Init_MissionRegion_sub_type mission(::ad_msgs::msg::MissionRegion::_mission_type arg)
  {
    msg_.mission = std::move(arg);
    return Init_MissionRegion_sub_type(msg_);
  }

private:
  ::ad_msgs::msg::MissionRegion msg_;
};

class Init_MissionRegion_radius
{
public:
  explicit Init_MissionRegion_radius(::ad_msgs::msg::MissionRegion & msg)
  : msg_(msg)
  {}
  Init_MissionRegion_mission radius(::ad_msgs::msg::MissionRegion::_radius_type arg)
  {
    msg_.radius = std::move(arg);
    return Init_MissionRegion_mission(msg_);
  }

private:
  ::ad_msgs::msg::MissionRegion msg_;
};

class Init_MissionRegion_y
{
public:
  explicit Init_MissionRegion_y(::ad_msgs::msg::MissionRegion & msg)
  : msg_(msg)
  {}
  Init_MissionRegion_radius y(::ad_msgs::msg::MissionRegion::_y_type arg)
  {
    msg_.y = std::move(arg);
    return Init_MissionRegion_radius(msg_);
  }

private:
  ::ad_msgs::msg::MissionRegion msg_;
};

class Init_MissionRegion_x
{
public:
  Init_MissionRegion_x()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_MissionRegion_y x(::ad_msgs::msg::MissionRegion::_x_type arg)
  {
    msg_.x = std::move(arg);
    return Init_MissionRegion_y(msg_);
  }

private:
  ::ad_msgs::msg::MissionRegion msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::ad_msgs::msg::MissionRegion>()
{
  return ad_msgs::msg::builder::Init_MissionRegion_x();
}

}  // namespace ad_msgs

#endif  // AD_MSGS__MSG__DETAIL__MISSION_REGION__BUILDER_HPP_
