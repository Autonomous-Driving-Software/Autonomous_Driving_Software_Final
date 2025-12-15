// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from ad_msgs:msg/PolyfitLaneDataArray.idl
// generated code does not contain a copyright notice

#ifndef AD_MSGS__MSG__DETAIL__POLYFIT_LANE_DATA_ARRAY__BUILDER_HPP_
#define AD_MSGS__MSG__DETAIL__POLYFIT_LANE_DATA_ARRAY__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "ad_msgs/msg/detail/polyfit_lane_data_array__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace ad_msgs
{

namespace msg
{

namespace builder
{

class Init_PolyfitLaneDataArray_polyfitlanes
{
public:
  explicit Init_PolyfitLaneDataArray_polyfitlanes(::ad_msgs::msg::PolyfitLaneDataArray & msg)
  : msg_(msg)
  {}
  ::ad_msgs::msg::PolyfitLaneDataArray polyfitlanes(::ad_msgs::msg::PolyfitLaneDataArray::_polyfitlanes_type arg)
  {
    msg_.polyfitlanes = std::move(arg);
    return std::move(msg_);
  }

private:
  ::ad_msgs::msg::PolyfitLaneDataArray msg_;
};

class Init_PolyfitLaneDataArray_frame_id
{
public:
  Init_PolyfitLaneDataArray_frame_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_PolyfitLaneDataArray_polyfitlanes frame_id(::ad_msgs::msg::PolyfitLaneDataArray::_frame_id_type arg)
  {
    msg_.frame_id = std::move(arg);
    return Init_PolyfitLaneDataArray_polyfitlanes(msg_);
  }

private:
  ::ad_msgs::msg::PolyfitLaneDataArray msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::ad_msgs::msg::PolyfitLaneDataArray>()
{
  return ad_msgs::msg::builder::Init_PolyfitLaneDataArray_frame_id();
}

}  // namespace ad_msgs

#endif  // AD_MSGS__MSG__DETAIL__POLYFIT_LANE_DATA_ARRAY__BUILDER_HPP_
