import argparse
from pathlib import Path
import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Детекція птахів на відео за допомогою YOLO26n.")
    parser.add_argument(
        "--input",
        type=str,
        default="examples/input.mp4",
        help="Шлях до вхідного відеофайлу"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="examples/output.mp4",
        help="Шлях для збереження обробленого відеофайлу"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/best.pt",
        help="Шлях до навчених ваг моделі (.pt)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.4,
        help="Поріг впевненості (confidence threshold) для відображення детекцій"
    )
    return parser.parse_args()


def process_video(input_path: Path, output_path: Path, weights_path: Path, conf_threshold: float) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Вхідний відеофайл не знайдено: {input_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Файл ваг моделі не знайдено: {weights_path}")

    # Створюємо папку для вихідного відео, якщо її ще немає
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Завантаження моделі з вагами: {weights_path}...")
    model = YOLO(str(weights_path))

    # Відкриваємо відеопотік через OpenCV
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Не вдалося відкрити відеофайл: {input_path}")

    # Отримуємо параметри вхідного відео (ширина, висота, FPS, загальна кількість кадрів)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Налаштовуємо запис вихідного відео у форматі MP4
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    print(f"Початок обробки відео ({total_frames} кадрів, {width}x{height} @ {fps:.1f} FPS)...")
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            # Запускаємо детекцію на поточному кадрі
            results = model.predict(source=frame, conf=conf_threshold, verbose=False)

            # Метод plot() автоматично малює bounding box, назву класу та confidence
            annotated_frame = results[0].plot()

            # Записуємо оброблений кадр у вихідний файл
            out.write(annotated_frame)

            if frame_idx % 50 == 0 or frame_idx == total_frames:
                print(f"Оброблено кадрів: {frame_idx}/{total_frames}")
    finally:
        # Гарантовано звільняємо ресурси навіть у разі помилки
        cap.release()
        out.release()

    print(f"\n[УСПІХ] Обробку завершено! Відео збережено у: {output_path}")


def main() -> None:
    args = parse_args()
    process_video(
        input_path=Path(args.input),
        output_path=Path(args.output),
        weights_path=Path(args.weights),
        conf_threshold=args.conf
    )


if __name__ == "__main__":
    main()
