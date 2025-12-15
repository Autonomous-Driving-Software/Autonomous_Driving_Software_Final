// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from ad_msgs:msg/LanePointDataArray.idl
// generated code does not contain a copyright notice

#ifndef AD_MSGS__MSG__DETAIL__LANE_POINT_DATA_ARRAY__BUILDER_HPP_
#define AD_MSGS__MSG__DETAIL__LANE_POINT_DATA_ARRAY__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "ad_msgs/msg/detail/lane_point_data_array__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace ad_msgs
{

namespace msg
{

namespace builder
{

class Init_LanePointDataArray_lane
{
public:
  explicit Init_LanePointDataArray_lane(::ad_msgs::msg::LanePointDataArray & msg)
  : msg_(msg)
  {}
  ::ad_msgs::msg::LanePointDataArray lane(::ad_msgs::msg::LanePointDataArray::_lane_type arg)
  {
    msg_.lane = std::move(arg);
    return std::move(msg_);
  }

private:
  ::ad_msgs::msg::LanePointDataArray msg_;
};

class Init_LanePointDataArray_id
{
public:
  explicit Init_LanePointDataArray_id(::ad_msgs::msg::LanePointDataArray & msg)
  : msg_(msg)
  {}
  Init_LanePointDataArray_lane id(::ad_msgs::msg::LanePointDataArray::_id_type arg)
  {
    msg_.id = std::move(arg);
    return Init_LanePointDataArray_lane(msg_);
  }

private:
  ::ad_msgs::msg::LanePointDataArray msg_;
};

class Init_LanePointDataArray_frame_id
{
public:
  Init_LanePointDataArray_frame_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_LanePointDataArray_id frame_id(::ad_msgs::msg::LanePointDataArray::_frame_id_type arg)
  {
    msg_.frame_id = std::move(arg);
    return Init_LanePointDataArray_id(msg_);
  }

private:
  ::ad_msgs::msg::LanePointDataArray msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::ad_msgs::msg::LanePointDataArray>()
{
  return ad_msgs::msg::builder::Init_LanePointDataArray_frame_id();
}

}  // namespace ad_msgs

#endif  // AD_MSGS__MSG__DETAIL__LANE_POINT_DATA_ARRAY__BUILDER_HPP_
