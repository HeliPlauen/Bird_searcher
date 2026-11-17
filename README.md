

Запуск встановлення залежностей (на локальному коимп'ютері):

pip install -r requirements.txt

---

Команда для запуску скрипта навчання (тестова, на локальному коимп'ютері):

python train.py --epochs 1 --batch 8

---

Команда для запуску скрипта навчання (робоча):

python train.py 

---

Компанда для для обробки відео навченою моделлю:

python predict_video.py —input examples/input.mp4 —output examples/output.mp4

---

