# SPDX-License-Identifier: MIT

import sys
from unittest.mock import MagicMock, patch

import pytest
import numpy as np

import jaxonomy
from jaxonomy.testing import requires_jax

# Mock out rclpy before importing anything from jaxonomy.library.ros2
mock_rclpy = MagicMock()
mock_rclpy.ok.return_value = True
sys.modules["rclpy"] = mock_rclpy

from jaxonomy.library.ros2 import Ros2Publisher, Ros2Subscriber, _fixup_dtype

class MockTwist:
    def __init__(self):
        self.linear = MagicMock()
        self.angular = MagicMock()
        self.linear.x = 0.0
        self.angular.z = 0.0

@requires_jax()
@patch("jaxonomy.library.ros2.rclpy", mock_rclpy)
def test_ros2_publisher():
    dt = 0.1
    topic = "/turtle1/cmd_vel"
    msg_type = MockTwist
    fields = {"linear.x": float, "angular.z": float}
    
    # Mock node creation
    mock_node = MagicMock()
    mock_publisher = MagicMock()
    mock_node.create_publisher.return_value = mock_publisher
    mock_rclpy.create_node.return_value = mock_node
    
    pub = Ros2Publisher(dt=dt, topic=topic, msg_type=msg_type, fields=fields)
    
    # Check proper dtype translation
    assert pub.input_types[0] == np.float64
    assert pub.input_types[1] == np.float64
    
    assert pub.node == mock_node
    assert pub.publisher == mock_publisher

    # Call the internal publish method directly to bypass io_callback and pure execution
    pub._publish_message(2.0, 3.0)
    
    # Verify the message was published via the mock
    mock_publisher.publish.assert_called()
    published_msg = mock_publisher.publish.call_args[0][0]
    
    assert isinstance(published_msg, MockTwist)
    assert published_msg.linear.x == 2.0
    assert published_msg.angular.z == 3.0
    
    # Terminate and verify cleanup
    pub.post_simulation_finalize()
    mock_node.destroy_publisher.assert_called_with(mock_publisher)
    mock_node.destroy_node.assert_called()


@requires_jax()
@patch("jaxonomy.library.ros2.rclpy", mock_rclpy)
def test_ros2_subscriber():
    dt = 0.1
    topic = "/turtle1/pose"
    msg_type = MockTwist
    fields = {"linear.x": float, "angular.z": float}
    
    # Mock node creation
    mock_node = MagicMock()
    mock_subscription = MagicMock()
    mock_node.create_subscription.return_value = mock_subscription
    mock_rclpy.create_node.return_value = mock_node
    
    sub = Ros2Subscriber(dt=dt, topic=topic, msg_type=msg_type, fields=fields)
    
    # Check initialization
    assert sub.node == mock_node
    assert sub.subscription == mock_subscription
    
    # Capture callback
    ros2_callback = mock_node.create_subscription.call_args[0][2]
    
    # Dispatch a mocked message from ROS2
    incoming_msg = MockTwist()
    incoming_msg.linear.x = 4.5
    incoming_msg.angular.z = -1.2
    
    ros2_callback(incoming_msg)
    
    # Check internal state of subscriber directly to bypass io_callback
    assert sub._last_msg == incoming_msg
    
    # Terminate and verify cleanup
    sub.post_simulation_finalize()
    mock_node.destroy_subscription.assert_called_with(mock_subscription)
    mock_node.destroy_node.assert_called()

def test_fixup_dtype():
    # String conversions
    assert _fixup_dtype("float") == np.float64
    assert _fixup_dtype("int") == np.int64
    assert _fixup_dtype("bool") == np.bool_
    
    # Type objects
    assert _fixup_dtype(float) == np.float64
    assert _fixup_dtype(int) == np.int64
    assert _fixup_dtype(bool) == np.bool_
    
    # Existing numpy types
    assert _fixup_dtype(np.float32) == np.float32

    with pytest.raises(ValueError):
        _fixup_dtype("unknown_dtype")

    with pytest.raises(ValueError):
        _fixup_dtype(123)
