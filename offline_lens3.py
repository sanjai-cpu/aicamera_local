# Optimized BLIP-based offline lens script
# - Resizes input to 384x384 (BLIP's native size)
# - Uses beam search for better accuracy
# - Automatically uses GPU if available

import cv2
from PIL import Image
from transformers import pipeline
import torch

def main():
    print("Initializing webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Detect device: GPU if available, otherwise CPU
    device = 0 if torch.cuda.is_available() else -1
    print(f"⏳ Loading BLIP model on {'GPU' if device == 0 else 'CPU'}...")

    try:
        image_to_text_pipeline = pipeline(
            "image-to-text",
            model="Salesforce/blip-image-captioning-base",
            device=device
        )
        print("✅ Model loaded successfully.")
        print("\nPress 's' to capture an image and get a description.")
        print("Press 'q' to quit the application.")
    except Exception as e:
        print(f"❌ Failed to load the model. Details: {e}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to capture image from webcam.")
            break

        cv2.imshow("Live Webcam Feed (s=capture, q=quit)", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            print("\n📸 Capturing image...")
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Resize to BLIP’s preferred input size
            pil_img = pil_img.resize((384, 384))

            try:
                result = image_to_text_pipeline(
                    pil_img,
                    generate_kwargs={
                        "max_new_tokens": 50,
                        "num_beams": 5,
                        "do_sample": False
                    }
                )
                description = result[0]["generated_text"]
                print("📝 Description:", description)
            except Exception as e:
                print(f"❌ Inference error: {e}")

        elif key == ord("q"):
            print("👋 Quitting...")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    """
    Setup:
    pip install torch torchvision torchaudio
    pip install transformers
    pip install opencv-python Pillow accelerate
    """
    main()
dscssvvs