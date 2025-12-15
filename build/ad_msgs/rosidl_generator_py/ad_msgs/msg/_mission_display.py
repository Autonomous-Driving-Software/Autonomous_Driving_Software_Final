# generated from rosidl_generator_py/resource/_idl.py.em
# with input from ad_msgs:msg/MissionDisplay.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_MissionDisplay(type):
    """Metaclass of message 'MissionDisplay'."""

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
                'ad_msgs.msg.MissionDisplay')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__mission_display
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__mission_display
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__mission_display
            cls._TYPE_SUPPORT = module.type_support_msg__msg__mission_display
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__mission_display

            from ad_msgs.msg import MissionObject
            if MissionObject.__class__._TYPE_SUPPORT is None:
                MissionObject.__class__.__import_type_support__()

            from ad_msgs.msg import MissionRegion
            if MissionRegion.__class__._TYPE_SUPPORT is None:
                MissionRegion.__class__.__import_type_support__()

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class MissionDisplay(metaclass=Metaclass_MissionDisplay):
    """Message class 'MissionDisplay'."""

    __slots__ = [
        '_objects',
        '_regions',
    ]

    _fields_and_field_types = {
        'objects': 'sequence<ad_msgs/MissionObject>',
        'regions': 'sequence<ad_msgs/MissionRegion>',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.NamespacedType(['ad_msgs', 'msg'], 'MissionObject')),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.NamespacedType(['ad_msgs', 'msg'], 'MissionRegion')),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.objects = kwargs.get('objects', [])
        self.regions = kwargs.get('regions', [])

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
        if self.objects != other.objects:
            return False
        if self.regions != other.regions:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def objects(self):
        """Message field 'objects'."""
        return self._objects

    @objects.setter
    def objects(self, value):
        if __debug__:
            from ad_msgs.msg import MissionObject
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 all(isinstance(v, MissionObject) for v in value) and
                 True), \
                "The 'objects' field must be a set or sequence and each value of type 'MissionObject'"
        self._objects = value

    @builtins.property
    def regions(self):
        """Message field 'regions'."""
        return self._regions

    @regions.setter
    def regions(self, value):
        if __debug__:
            from ad_msgs.msg import MissionRegion
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 all(isinstance(v, MissionRegion) for v in value) and
                 True), \
                "The 'regions' field must be a set or sequence and each value of type 'MissionRegion'"
        self._regions = value
