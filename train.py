import argparse
import shutil
from pathlib import Path
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Донавчання моделі YOLO26n для детекції птахів.")
    parser.add_argument(
        "--data",
        type=str,
        default="bird.v7i.yolo26_1014/data.yaml",
        help="Шлях до конфігураційного файлу датасету (data.yaml)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
        help="Кількість епох навчання (для швидкого тесту на CPU вкажіть 1)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Розмір вхідного зображення"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Розмір пакету (batch size)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)

    if not data_path.exists():
        raise FileNotFoundError(f"Файл конфігурації датасету не знайдено за шляхом: {data_path}")

    print(f"Завантаження попередньо навченої моделі YOLO26n...")
    model = YOLO("yolo26n.pt")

    print(f"Старт навчання на {args.epochs} епох(у) з датасетом {data_path}...")
    # Запускаємо тренування. Ultralytics автоматично визначить GPU, якщо вона є, або використає CPU.
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project="runs/detect",
        name="bird_train",
        exist_ok=True  # Перезаписувати папку при повторних запусках
    )

    # Автоматично копіюємо найкращі ваги у цільову теку weights/best.pt згідно з ТЗ
    best_weights_src = Path(results.save_dir) / "weights" / "best.pt"
    weights_dir = Path("weights")
    weights_dir.mkdir(parents=True, exist_ok=True)
    best_weights_dst = weights_dir / "best.pt"

    if best_weights_src.exists():
        shutil.copy2(best_weights_src, best_weights_dst)
        print(f"\n[УСПІХ] Найкращі ваги успішно збережено у: {best_weights_dst}")
    else:
        print(f"\n[УВАГА] Файл {best_weights_src} не знайдено після завершення навчання.")


if __name__ == "__main__":
    main()
