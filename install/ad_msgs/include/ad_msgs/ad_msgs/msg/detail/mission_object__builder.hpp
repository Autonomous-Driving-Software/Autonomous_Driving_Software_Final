// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from ad_msgs:msg/MissionObject.idl
// generated code does not contain a copyright notice

#ifndef AD_MSGS__MSG__DETAIL__MISSION_OBJECT__BUILDER_HPP_
#define AD_MSGS__MSG__DETAIL__MISSION_OBJECT__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "ad_msgs/msg/detail/mission_object__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace ad_msgs
{

namespace msg
{

namespace builder
{

class Init_MissionObject_velocity
{
public:
  explicit Init_MissionObject_velocity(::ad_msgs::msg::MissionObject & msg)
  : msg_(msg)
  {}
  ::ad_msgs::msg::MissionObject velocity(::ad_msgs::msg::MissionObject::_velocity_type arg)
  {
    msg_.velocity = std::move(arg);
    return std::move(msg_);
  }

private:
  ::ad_msgs::msg::MissionObject msg_;
};

class Init_MissionObject_yaw
{
public:
  explicit Init_MissionObject_yaw(::ad_msgs::msg::MissionObject & msg)
  : msg_(msg)
  {}
  Init_MissionObject_velocity yaw(::ad_msgs::msg::MissionObject::_yaw_type arg)
  {
    msg_.yaw = std::move(arg);
    return Init_MissionObject_velocity(msg_);
  }

private:
  ::ad_msgs::msg::MissionObject msg_;
};

class Init_MissionObject_y
{
public:
  explicit Init_MissionObject_y(::ad_msgs::msg::MissionObject & msg)
  : msg_(msg)
  {}
  Init_MissionObject_yaw y(::ad_msgs::msg::MissionObject::_y_type arg)
  {
    msg_.y = std::move(arg);
    return Init_MissionObject_yaw(msg_);
  }

private:
  ::ad_msgs::msg::MissionObject msg_;
};

class Init_MissionObject_x
{
public:
  explicit Init_MissionObject_x(::ad_msgs::msg::MissionObject & msg)
  : msg_(msg)
  {}
  Init_MissionObject_y x(::ad_msgs::msg::MissionObject::_x_type arg)
  {
    msg_.x = std::move(arg);
    return Init_MissionObject_y(msg_);
  }

private:
  ::ad_msgs::msg::MissionObject msg_;
};

class Init_MissionObject_is_reach_end
{
public:
  explicit Init_MissionObject_is_reach_end(::ad_msgs::msg::MissionObject & msg)
  : msg_(msg)
  {}
  Init_MissionObject_x is_reach_end(::ad_msgs::msg::MissionObject::_is_reach_end_type arg)
  {
    msg_.is_reach_end = std::move(arg);
    return Init_MissionObject_x(msg_);
  }

private:
  ::ad_msgs::msg::MissionObject msg_;
};

class Init_MissionObject_object_type
{
public:
  Init_MissionObject_object_type()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_MissionObject_is_reach_end object_type(::ad_msgs::msg::MissionObject::_object_type_type arg)
  {
    msg_.object_type = std::move(arg);
    return Init_MissionObject_is_reach_end(msg_);
  }

private:
  ::ad_msgs::msg::MissionObject msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::ad_msgs::msg::MissionObject>()
{
  return ad_msgs::msg::builder::Init_MissionObject_object_type();
}

}  // namespace ad_msgs

#endif  // AD_MSGS__MSG__DETAIL__MISSION_OBJECT__BUILDER_HPP_
