// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from ad_msgs:msg/Mission.idl
// generated code does not contain a copyright notice

#ifndef AD_MSGS__MSG__DETAIL__MISSION__BUILDER_HPP_
#define AD_MSGS__MSG__DETAIL__MISSION__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "ad_msgs/msg/detail/mission__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace ad_msgs
{

namespace msg
{

namespace builder
{

class Init_Mission_parking
{
public:
  explicit Init_Mission_parking(::ad_msgs::msg::Mission & msg)
  : msg_(msg)
  {}
  ::ad_msgs::msg::Mission parking(::ad_msgs::msg::Mission::_parking_type arg)
  {
    msg_.parking = std::move(arg);
    return std::move(msg_);
  }

private:
  ::ad_msgs::msg::Mission msg_;
};

class Init_Mission_speed_limit
{
public:
  explicit Init_Mission_speed_limit(::ad_msgs::msg::Mission & msg)
  : msg_(msg)
  {}
  Init_Mission_parking speed_limit(::ad_msgs::msg::Mission::_speed_limit_type arg)
  {
    msg_.speed_limit = std::move(arg);
    return Init_Mission_parking(msg_);
  }

private:
  ::ad_msgs::msg::Mission msg_;
};

class Init_Mission_road_slope
{
public:
  explicit Init_Mission_road_slope(::ad_msgs::msg::Mission & msg)
  : msg_(msg)
  {}
  Init_Mission_speed_limit road_slope(::ad_msgs::msg::Mission::_road_slope_type arg)
  {
    msg_.road_slope = std::move(arg);
    return Init_Mission_speed_limit(msg_);
  }

private:
  ::ad_msgs::msg::Mission msg_;
};

class Init_Mission_road_condition
{
public:
  explicit Init_Mission_road_condition(::ad_msgs::msg::Mission & msg)
  : msg_(msg)
  {}
  Init_Mission_road_slope road_condition(::ad_msgs::msg::Mission::_road_condition_type arg)
  {
    msg_.road_condition = std::move(arg);
    return Init_Mission_road_slope(msg_);
  }

private:
  ::ad_msgs::msg::Mission msg_;
};

class Init_Mission_objects
{
public:
  Init_Mission_objects()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_Mission_road_condition objects(::ad_msgs::msg::Mission::_objects_type arg)
  {
    msg_.objects = std::move(arg);
    return Init_Mission_road_condition(msg_);
  }

private:
  ::ad_msgs::msg::Mission msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::ad_msgs::msg::Mission>()
{
  return ad_msgs::msg::builder::Init_Mission_objects();
}

}  // namespace ad_msgs

#endif  // AD_MSGS__MSG__DETAIL__MISSION__BUILDER_HPP_
