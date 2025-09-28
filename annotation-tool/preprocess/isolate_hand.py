from ultralytics import YOLO
import cv2
import numpy as np
import os

# Load a model
model = YOLO("yolo11n-seg.pt")  # load an official model

images = os.listdir("../../evaluation-test-sets/test-data-1/image_dir")  # add your image directory here
images = [os.path.join("../../evaluation-test-sets/test-data-1/image_dir", img) for img in images if img.endswith('.jpg')]

# Predict with the model
results = model(images)  # return a list of Results objects

for i, result in enumerate(results):

    image = cv2.imread(images[i])
    image_name = os.path.splitext(os.path.basename(images[i]))[0]
    output_dir = "../../evaluation-test-sets/test-data-1/output_dir"
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, f"{image_name}.jpg")
    h, w = image.shape[:2]

        # --- Build a combined person mask at original size ---
    if result.masks is not None:
        combined_mask = np.zeros((h, w), dtype=np.uint8)

        # Safer variable name to avoid shadowing outer 'i'
        for j, (mask_t, box) in enumerate(zip(result.masks.data, result.boxes)):
            # keep only 'person' (class 0)
            cls_id = int(box.cls[0].item()) if hasattr(box.cls[0], "item") else int(box.cls[0])
            if cls_id != 0:
                continue

            # to numpy (binary 0/1) and resize to original image size
            m = mask_t.detach().cpu().numpy()
            if m.ndim == 3:  # (1,H,W) → (H,W)
                m = m[0]
            m = (m > 0.5).astype(np.uint8)
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)

            combined_mask = np.maximum(combined_mask, m)

        if combined_mask.sum() == 0:
            print(f"No person masks kept for {image_name}")
            # still write an all-black image to be explicit, or skip — your call
            masked_image = np.zeros_like(image)
            cv2.imwrite(output_filename, masked_image)
            continue

        # --- Clean the mask: close gaps, fill holes, smooth edges ---
        # Close small gaps/edge noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        closed = cv2.morphologyEx(combined_mask * 255, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Fill holes via flood-fill from the border
        flood = closed.copy()
        ff_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)  # required by floodFill
        cv2.floodFill(flood, ff_mask, (0, 0), 255)          # fill background connected to border
        holes = cv2.bitwise_not(flood)                      # now holes are white
        filled = cv2.bitwise_or(closed, holes)              # add holes to the closed mask

        # Optional gentle edge smoothing (binary → slight blur → re-threshold)
        sm = cv2.GaussianBlur(filled, (5, 5), 0)
        _, clean_mask = cv2.threshold(sm, 127, 255, cv2.THRESH_BINARY)

        # --- Apply the cleaned mask (fully black background) ---
        mask_3c = np.repeat((clean_mask // 255)[:, :, None], 3, axis=2)
        masked_image = cv2.bitwise_and(image, mask_3c * 255)

        cv2.imwrite(output_filename, masked_image)
    else:
        cv2.imwrite(output_filename, image)
    print(f"Saved {output_filename}")
