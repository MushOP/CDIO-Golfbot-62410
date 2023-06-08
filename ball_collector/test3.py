import cv2
import numpy as np

balls = []

hsv_values = {'hmin': 0, 'smin': 0, 'vmin': 218,
              'hmax': 179, 'smax': 67, 'vmax': 255}

orange_hsv_values = {'hmin': 0, 'smin': 100, 'vmin': 100,
                     'hmax': 30, 'smax': 255, 'vmax': 255}

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    hmin, smin, vmin = hsv_values['hmin'], hsv_values['smin'], hsv_values['vmin']
    hmax, smax, vmax = hsv_values['hmax'], hsv_values['smax'], hsv_values['vmax']

    lower_range = np.array([hmin, smin, vmin])
    upper_range = np.array([hmax, smax, vmax])

    mask = cv2.inRange(hsv, lower_range, upper_range)

    orange_mask = cv2.inRange(hsv, np.array([orange_hsv_values['hmin'], orange_hsv_values['smin'], orange_hsv_values['vmin']]),
                              np.array([orange_hsv_values['hmax'], orange_hsv_values['smax'], orange_hsv_values['vmax']]))

    # Apply color overlay on orange mask
    orange_overlay = cv2.bitwise_and(frame, frame, mask=orange_mask)

    final_mask = mask + orange_mask

    contours, _ = cv2.findContours(
        final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:
                (x, y, w, h) = cv2.boundingRect(contour)
                radius = int(w / 2)
                if radius > 9 or radius < 7:
                    continue
                cv2.circle(frame, (int(x + w / 2), int(y + h / 2)),
                           int(max(w, h) / 2), (0, 255, 0), 2)
                balls.append((x + radius / 2, y + radius / 2, radius, len(balls)))

                # Add coordinates on the detected white balls
                text = f"({int(x + w / 2)}, {int(y + h / 2)})"
                cv2.putText(frame, text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Frame', frame)
    cv2.imshow('Orange Overlay', orange_overlay)
    cv2.imshow('Mask', final_mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
