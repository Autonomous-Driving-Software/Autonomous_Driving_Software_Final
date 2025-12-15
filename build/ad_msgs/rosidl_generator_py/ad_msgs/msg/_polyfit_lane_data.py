# generated from rosidl_generator_py/resource/_idl.py.em
# with input from ad_msgs:msg/PolyfitLaneData.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import math  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_PolyfitLaneData(type):
    """Metaclass of message 'PolyfitLaneData'."""

    _CREATE_ROS_MESSAGE = None
    _CONVERT_FROM_PY = None
    _CONVERT_TO_PY = None
    _DESTROY_ROS_MESSAGE = None
    _TYPE_SUPPORT = None

    __constants = {
    }

    @classmethod
    def __import_type_support__(cls):
        try:
            from rosidl_generator_py import import_type_support
            module = import_type_support('ad_msgs')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'ad_msgs.msg.PolyfitLaneData')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__polyfit_lane_data
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__polyfit_lane_data
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__polyfit_lane_data
            cls._TYPE_SUPPORT = module.type_support_msg__msg__polyfit_lane_data
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__polyfit_lane_data

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class PolyfitLaneData(metaclass=Metaclass_PolyfitLaneData):
    """Message class 'PolyfitLaneData'."""

    __slots__ = [
        '_frame_id',
        '_id',
        '_a0',
        '_a1',
        '_a2',
        '_a3',
    ]

    _fields_and_field_types = {
        'frame_id': 'string',
        'id': 'string',
        'a0': 'double',
        'a1': 'double',
        'a2': 'double',
        'a3': 'double',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.UnboundedString(),  # noqa: E501
        rosidl_parser.definition.UnboundedString(),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.frame_id = kwargs.get('frame_id', str())
        self.id = kwargs.get('id', str())
        self.a0 = kwargs.get('a0', float())
        self.a1 = kwargs.get('a1', float())
        self.a2 = kwargs.get('a2', float())
        self.a3 = kwargs.get('a3', float())

    def __repr__(self):
        typename = self.__class__.__module__.split('.')
        typename.pop()
        typename.append(self.__class__.__name__)
        args = []
        for s, t in zip(self.__slots__, self.SLOT_TYPES):
            field = getattr(self, s)
            fieldstr = repr(field)
            # We use Python array type for fields that can be directly stored
            # in them, and "normal" sequences for everything else.  If it is
            # a type that we store in an array, strip off the 'array' portion.
            if (
                isinstance(t, rosidl_parser.definition.AbstractSequence) and
                isinstance(t.value_type, rosidl_parser.definition.BasicType) and
                t.value_type.typename in ['float', 'double', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64']
            ):
                if len(field) == 0:
                    fieldstr = '[]'
                else:
                    assert fieldstr.startswith('array(')
                    prefix = "array('X', "
                    suffix = ')'
                    fieldstr = fieldstr[len(prefix):-len(suffix)]
            args.append(s[1:] + '=' + fieldstr)
        return '%s(%s)' % ('.'.join(typename), ', '.join(args))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if self.frame_id != other.frame_id:
            return False
        if self.id != other.id:
            return False
        if self.a0 != other.a0:
            return False
        if self.a1 != other.a1:
            return False
        if self.a2 != other.a2:
            return False
        if self.a3 != other.a3:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def frame_id(self):
        """Message field 'frame_id'."""
        return self._frame_id

    @frame_id.setter
    def frame_id(self, value):
        if __debug__:
            assert \
                isinstance(value, str), \
                "The 'frame_id' field must be of type 'str'"
        self._frame_id = value

    @builtins.property  # noqa: A003
    def id(self):  # noqa: A003
        """Message field 'id'."""
        return self._id

    @id.setter  # noqa: A003
    def id(self, value):  # noqa: A003
        if __debug__:
            assert \
                isinstance(value, str), \
                "The 'id' field must be of type 'str'"
        self._id = value

    @builtins.property
    def a0(self):
        """Message field 'a0'."""
        return self._a0

    @a0.setter
    def a0(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'a0' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'a0' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._a0 = value

    @builtins.property
    def a1(self):
        """Message field 'a1'."""
        return self._a1

    @a1.setter
    def a1(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'a1' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'a1' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._a1 = value

    @builtins.property
    def a2(self):
        """Message field 'a2'."""
        return self._a2

    @a2.setter
    def a2(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'a2' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'a2' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._a2 = value

    @builtins.property
    def a3(self):
        """Message field 'a3'."""
        return self._a3

    @a3.setter
    def a3(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'a3' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'a3' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._a3 = value
