"""
Real-time drowsiness detection — run after training.

Usage:
    python detect.py                              # Webcam mode
    python detect.py --video path.mp4             # Video file mode
    python detect.py --video path.mp4 --output result.mp4
    python detect.py --multimodal                 # Enable blink + head pose signals
"""
import argparse
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from src.engine.inference import DrowsinessDetector


def main():
    parser = argparse.ArgumentParser(description="Driver Drowsiness Detection")
    parser.add_argument("--video", type=str, default=None,
                        help="Path to input video (default: webcam)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save output video")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to trained model (.keras)")
    parser.add_argument("--multimodal", action="store_true",
                        help="Enable multimodal signals (blink rate + head pose)")
    args = parser.parse_args()

    detector = DrowsinessDetector(model_path=args.model)

    if args.multimodal:
        from src.models.multimodal import MultimodalFatigueAssessor
        detector.multimodal = MultimodalFatigueAssessor()
        print("  Multimodal mode: eye state + PERCLOS + head pose + blink rate")

    if args.video:
        summary = detector.run_video(args.video, args.output)
        if summary:
            print("\nSession Summary:")
            for k, v in summary.items():
                print(f"  {k}: {v}")
    else:
        detector.run_webcam()


if __name__ == "__main__":
    main()
