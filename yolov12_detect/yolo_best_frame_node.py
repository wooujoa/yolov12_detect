import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import supervision as sv
import cv2


class YoloBestFrameNode(Node):
    def __init__(self):
        super().__init__('yolo_best_frame_node')

        self.image_topic = '/camera/camera/color/image_raw'
        self.subscription = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            10
        )

        self.bridge = CvBridge()
        self.model = YOLO("/home/jwg/inference/best.pt")
        self.device = 'cpu'

        self.frame_buffer = []
        self.detection_buffer = []
        self.max_frames = 40
        self.frame_count = 0
        self.got_image = False

        self.get_logger().info("YOLO Best Frame Node Started")
        self.get_logger().info(f"Subscribing to image topic: {self.image_topic}")

        # 타이머로 토픽 상태 체크 (5초마다)
        self.timer = self.create_timer(5.0, self.check_topic_status)

    def check_topic_status(self):
        if not self.got_image:
            self.get_logger().warn(f"아직 이미지 메시지를 수신하지 못했습니다. '{self.image_topic}' 토픽이 퍼블리시되고 있는지 확인하세요.")
        else:
            self.get_logger().info("이미지 메시지 수신 중입니다. 계속 추론합니다.")

    def image_callback(self, msg):
        if self.frame_count >= self.max_frames:
            return

        self.got_image = True  # 이미지 한 번이라도 들어왔음

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")
            return

        try:
            results = self.model(frame, imgsz=640, device=self.device)[0]
        except Exception as e:
            self.get_logger().error(f"YOLO 추론 실패: {e}")
            return

        try:
            detections = sv.Detections.from_ultralytics(results)
        except Exception as e:
            self.get_logger().error(f"Detection 변환 실패: {e}")
            return

        self.frame_buffer.append(frame.copy())
        self.detection_buffer.append(detections)
        self.frame_count += 1

        self.get_logger().info(f"수신된 프레임: {self.frame_count}/{self.max_frames}")

        if self.frame_count == self.max_frames:
            self.process_best_frame()

    def process_best_frame(self):
        max_objects = 0
        best_index = -1

        for i, det in enumerate(self.detection_buffer):
            if len(det) > max_objects:
                max_objects = len(det)
                best_index = i

        if best_index == -1:
            self.get_logger().warn("감지된 객체가 없습니다.")
            return

        best_frame = self.frame_buffer[best_index]
        best_detections = self.detection_buffer[best_index]
        best_labels = [self.model.names[int(cls)] for cls in best_detections.class_id]

        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()

        annotated = box_annotator.annotate(scene=best_frame.copy(), detections=best_detections)
        annotated = label_annotator.annotate(scene=annotated, detections=best_detections, labels=best_labels)

        filename = "best_detection_frame.jpg"
        cv2.imwrite(filename, annotated)
        self.get_logger().info(f"최고 프레임 저장 완료: {filename}")


def main(args=None):
    rclpy.init(args=args)
    node = YoloBestFrameNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
