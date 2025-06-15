import argparse
import cv2
from padel.video import VideoSource
from padel.calibration import CourtCalibrator
from padel.ball import BallDetector
from padel.validator import FrameValidator
from padel.predictor import TrajectoryPredictor


def parse_args():
    parser = argparse.ArgumentParser(description="Padel HawkEye demo")
    parser.add_argument("--video", help="Path or URL of video source", default=0)
    parser.add_argument("--points", help="Initial calibration points x1,y1,x2,y2,x3,y3,x4,y4", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    src = VideoSource(args.video)

    ref_points = None
    if args.points:
        nums = [float(x) for x in args.points.split(',')]
        ref_points = [(nums[0], nums[1]), (nums[2], nums[3]),
                      (nums[4], nums[5]), (nums[6], nums[7])]
    calibrator = CourtCalibrator(reference_points=ref_points)
    detector = BallDetector()
    validator = FrameValidator()
    predictor = TrajectoryPredictor()

    while True:
        frame = src.read()
        if frame is None:
            break

        if not validator.is_valid(frame):
            cv2.putText(frame, "Invalid view", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2)
            cv2.imshow("Padel", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        if calibrator.homography is None:
            calibrator.calibrate(frame)

        ball_pos = detector.detect(frame)
        if ball_pos is not None:
            world = calibrator.pixel_to_world(ball_pos)
            predictor.add_point(world)
            cv2.circle(frame, ball_pos, 5, (0, 255, 0), -1)
            cv2.putText(frame, f"{world[0]:.2f}, {world[1]:.2f}",
                        (ball_pos[0]+10, ball_pos[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        prediction = predictor.predict()
        if prediction is not None and calibrator.homography is not None:
            # Convert world prediction back to pixel for visualization.
            inv_h = cv2.invert(calibrator.homography)[1]
            pt = calibrator.homography @ [[prediction[0]], [prediction[1]], [1]]
        trajectory_points = [calibrator.homography @ [[p[0]], [p[1]], [1]] for p in predictor.points]
        for i in range(1, len(trajectory_points)):
            p1 = trajectory_points[i-1]
            p2 = trajectory_points[i]
            p1 = (int(p1[0]/p1[2]), int(p1[1]/p1[2]))
            p2 = (int(p2[0]/p2[2]), int(p2[1]/p2[2]))
            cv2.line(frame, p1, p2, (255, 0, 0), 2)

        cv2.imshow("Padel", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    src.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
