# RealSense Camera Integration

### High-Level Architectural Constraint: The Client-Server Model

First, it's crucial to understand that the Jetson Orin, while powerful for an edge device, **cannot run the 7B parameter OpenVLA model locally**. The model requires a high-end GPU with significant VRAM (>=16-18 GB for inference).

Therefore, the only viable integration strategy is to leverage the **client-server architecture** demonstrated in the `experiments/aloha/` part of the codebase.

*   **Jetson Orin (Client):** Your Jetson will be responsible for capturing images from the Realsense camera, packaging them with a language instruction, and sending them over the network.
*   **Powerful PC/Server with GPU (Server):** A separate machine will run the `vla-scripts/deploy.py` script, which hosts the OpenVLA model and exposes a REST API endpoint for inference.

Your integration work will focus entirely on the **client-side application** running on the Jetson Orin. The server-side code (`deploy.py`) requires no changes, as it is designed to be hardware-agnostic, simply accepting NumPy arrays as image inputs.

Here are the two primary options for integrating the Realsense camera on the Jetson Orin.

---

### Option 1: Direct Python Integration with `pyrealsense2`

This is the most straightforward approach, ideal for rapid prototyping, self-contained scripts, and projects that don't require the complexity of a larger robotics framework.

*   **Best For:** Quick proofs-of-concept, single-robot applications, developers more comfortable with scripting than with robotics frameworks.
*   **Architectural Approach:** You will modify a client script (like `experiments/robot/aloha/run_aloha_eval.py`) to directly import the `pyrealsense2` library, capture frames in the main loop, and send them to the server.

```mermaid
graph TD
    subgraph Jetson Orin (Client)
        A[Realsense Camera] -->|USB| B(pyrealsense2 Library)
        B -->|NumPy Array| C{Custom Python Script<br>(Modified run_aloha_eval.py)}
        C -->|JSON Payload| D[HTTP Request (requests)]
    end

    subgraph GPU Server
        E[OpenVLA Server<br>(deploy.py)]
    end

    D -->|Network| E

    style A fill:#f8cecc,stroke:#b85450
    style E fill:#dae8fc,stroke:#6c8ebf
```

*   **Pros:**
    *   **Simple & Fast to Implement:** Requires minimal new dependencies and can be integrated into existing scripts quickly.
    *   **Low Overhead:** No extra processes or middleware; just a Python script.
    *   **Direct Control:** Full, low-level control over camera settings and stream management within your script.
*   **Cons:**
    *   **Less Scalable:** Becomes cumbersome if you need to add more sensors, actuators, or logic.
    *   **Not Standard Robotics Practice:** Lacks the modularity and introspection capabilities of frameworks like ROS.
    *   **Tightly Coupled:** The camera logic is tightly coupled with your application logic.

#### **Step-by-Step Implementation Guide (Option 1):**

1.  **Install Realsense SDK on Jetson:** Ensure you have the Intel RealSense SDK 2.0 (`librealsense`) and the Python wrapper (`pyrealsense2`) installed on your Jetson Orin.

2.  **Create a Camera Client Class:** To keep your code clean, create a helper class to manage the Realsense camera.

    ```python
    # in a new file, e.g., experiments/robot/realsense/camera_utils.py
    import pyrealsense2 as rs
    import numpy as np

    class RealsenseClient:
        def __init__(self, width=640, height=480, fps=30):
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
            self.pipeline.start(config)

        def get_observation(self):
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                return None

            # Convert to NumPy array
            color_image = np.asanyarray(color_frame.get_data())

            # The model expects RGB, but pyrealsense gives BGR
            color_image_rgb = color_image[:, :, ::-1]

            # Return in the format the VLA expects
            # NOTE: Add proprioception if you have it, e.g., from your robot's SDK
            return {
                "full_image": color_image_rgb,
                # "wrist_image": None, # Add this if you have a second camera
                "state": np.zeros(PROPRIO_DIM) # Placeholder for proprioceptive state
            }

        def stop(self):
            self.pipeline.stop()
    ```

3.  **Modify the Evaluation Script:** Adapt `experiments/robot/aloha/run_aloha_eval.py` to use your `RealsenseClient`.

    ```python
    # In your modified run_aloha_eval.py or a new script
    ...
    # from experiments.robot.aloha.aloha_utils import get_aloha_env ... # REMOVE THIS
    from experiments.robot.realsense.camera_utils import RealsenseClient # ADD THIS
    from experiments.robot.openvla_utils import get_action_from_server, resize_image_for_policy
    ...

    def run_episode(...):
        # ... setup ...

        # Instead of `env.get_observation()`, use your RealsenseClient
        # This code would be inside your main control loop
        observation_from_camera = camera_client.get_observation()
        if observation_from_camera is None:
            continue

        # Resize image and prepare payload for server
        img_resized = resize_image_for_policy(observation_from_camera["full_image"], resize_size)
        payload = {
            "full_image": img_resized,
            # "wrist_image": ..., # if you have one
            "state": observation_from_camera["state"],
            "instruction": task_description,
        }

        # Query model to get action
        actions = get_action_from_server(payload, server_endpoint)
        # ... execute action ...

    ...

    @draccus.wrap()
    def eval_realsense(cfg: GenerateConfig):
        ...
        # Instead of `env = get_aloha_env()`, initialize your camera
        camera_client = RealsenseClient()
        ...
        # In the main loop, call your modified run_episode
        run_episode(cfg, camera_client, ...)
        ...
        camera_client.stop()

    if __name__ == "__main__":
        eval_realsense()
    ```

---

### Option 2: ROS/ROS2 Integration

This is the industry-standard and most robust approach for robotics. It decouples the camera driver from your application logic, allowing for greater modularity and scalability.

*   **Best For:** Complex robotics projects, multi-sensor systems, teams, and systems that require high reliability and introspection.
*   **Architectural Approach:** A standard ROS node for the Realsense camera will publish images to a topic. Your Python client script will be a ROS node that subscribes to this topic to receive images.

```mermaid
graph TD
    subgraph Jetson Orin (Client)
        A[Realsense Camera] -->|USB| B[realsense-ros Node]
        B -- Publishes --> C((/camera/color/image_raw Topic))
        C -- Subscribes --> D{Your Python ROS Node<br>(Modified run_aloha_eval.py)}
        D -->|JSON Payload| E[HTTP Request (requests)]
    end

    subgraph GPU Server
        F[OpenVLA Server<br>(deploy.py)]
    end

    E -->|Network| F

    style A fill:#f8cecc,stroke:#b85450
    style F fill:#dae8fc,stroke:#6c8ebf
```

*   **Pros:**
    *   **Decoupled & Modular:** The camera driver is a separate process. You can restart your application code without restarting the camera.
    *   **Scalable:** Easy to add more sensors (e.g., another camera, an IMU) by simply subscribing to more topics.
    *   **Standardized:** Uses a well-known, powerful robotics framework. Tools like `rviz` and `ros2 topic echo` make debugging easy.
*   **Cons:**
    *   **Higher Initial Complexity:** Requires setting up a ROS/ROS2 workspace and understanding the publisher/subscriber model.
    *   **More Overhead:** Involves running the ROS master/daemon and multiple processes.

#### **Step-by-Step Implementation Guide (Option 2):**

1.  **Install ROS2 & Realsense Node:** Install ROS2 on your Jetson Orin and the `realsense-ros` package. Make sure it's running and publishing topics correctly. You can check with `ros2 topic list` and `ros2 topic echo /camera/color/image_raw`.

2.  **Create a ROS-based Client:** Modify your client script to be a ROS2 node.

    ```python
    # in a new file, e.g., experiments/robot/realsense/ros_client.py
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image as ROSImage
    from cv_bridge import CvBridge
    import numpy as np

    class RealsenseROSClient(Node):
        def __init__(self):
            super().__init__('openvla_client_node')
            self.subscription = self.create_subscription(
                ROSImage,
                '/camera/color/image_raw', # Or your specific Realsense topic
                self.image_callback,
                10)
            self.bridge = CvBridge()
            self.latest_image = None
            self.get_logger().info("ROS client node started, subscribing to image topic...")

        def image_callback(self, msg):
            # Convert ROS Image message to OpenCV image, then to RGB NumPy array
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.latest_image = cv_image[:, :, ::-1] # BGR to RGB

        def get_observation(self):
            if self.latest_image is None:
                return None
            return {
                "full_image": self.latest_image.copy(),
                "state": np.zeros(PROPRIO_DIM) # Placeholder for proprioceptive state
            }
    ```

3.  **Integrate into the Main Script:** Your main script will initialize `rclpy`, spin the node in a separate thread (or use an executor), and query the `RealsenseROSClient` for the latest image in the main loop.

    ```python
    # In your main evaluation script
    import rclpy
    ...
    from experiments.robot.realsense.ros_client import RealsenseROSClient
    ...

    @draccus.wrap()
    def eval_realsense_ros(cfg: GenerateConfig):
        rclpy.init()
        ros_client = RealsenseROSClient()

        # Spin the node in a background thread to keep callbacks running
        import threading
        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(ros_client)
        executor_thread = threading.Thread(target=executor.spin, daemon=True)
        executor_thread.start()

        # Wait for the first image
        while ros_client.get_observation() is None:
            print("Waiting for first image from ROS topic...")
            time.sleep(1)

        # Main evaluation loop...
        for episode_idx in ...:
            # In your loop, get observation from the ROS client
            observation_from_camera = ros_client.get_observation()
            # ... process and send to server as in Option 1 ...

        ros_client.destroy_node()
        rclpy.shutdown()

    if __name__ == "__main__":
        eval_realsense_ros()
    ```

### **Key Architectural & Implementation Considerations**

*   **Image Preprocessing:** Regardless of the option you choose, the image sent to the server *must* match the format expected by the model. The `openvla-oft` model expects a NumPy array of shape `(H, W, 3)` with `dtype=np.uint8`. Your client script is responsible for this. The `resize_image_for_policy` function in `experiments/robot/openvla_utils.py` is crucial here. The server expects the resized image.
*   **Multiple Cameras:** The OpenVLA architecture supports multiple camera inputs (e.g., `full_image`, `wrist_image`).
    *   **With `pyrealsense2`:** You can connect multiple cameras and manage multiple `rs.pipeline` objects, one for each camera's serial number.
    *   **With ROS:** You would launch multiple instances of the `realsense-ros` node, each in its own namespace, publishing to different topics (e.g., `/cam_1/color/image_raw`, `/cam_2/color/image_raw`). Your client node would subscribe to all of them.
*   **Network Performance:** Since you are sending images from the Jetson to a server, network latency and bandwidth are important. You may need to compress the images (e.g., JPEG encode) on the client before sending and decode them on the server to reduce bandwidth usage, although this adds computational overhead. For a local lab network, sending raw resized images is often fine.

### **Recommendation**

*   For a **quick start** or a simple project, **Option 1 (Direct Python Integration)** is the fastest way to get results.
*   For any project that you plan to build upon, scale, or collaborate on, **Option 2 (ROS/ROS2 Integration)** is the superior choice and the standard in the robotics community. It will save you significant time and effort in the long run.