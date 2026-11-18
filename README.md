# Ручное и искусственное обогащение данных: SMS Spam и Clotho

Курсовая работа по анализу данных (МАИ, 2 курс). Работа выполняется в паре.

**Тема:** сравнение того, как ручное и искусственное обогащение данных влияет на качество
и устойчивость моделей машинного обучения.

**Контрольная точка 18.11.2026:** [docs/checkpoint_18_11.md](docs/checkpoint_18_11.md) — что сделано, результаты разметки, ссылки на всю документацию.

Датасеты с разметкой хранятся на Google Диске (ссылка в файле контрольной точки), а не в репозитории.

## Датасеты

| | SMS Spam Collection | Clotho v2.1 |
|---|---|---|
| Источник | [UCI #228](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) | [Zenodo 4783391](https://zenodo.org/records/4783391) |
| Модальность | текст | аудио + текст |
| Объём | 5 574 сообщения | 6 974 аудиозаписи, 5 подписей к каждой |
| Задача | бинарная классификация (ham / spam) | audio captioning: по звуку сгенерировать текстовое описание |
| Лицензия | CC BY 4.0 | аудио — лицензии Freesound; подписи — условия Tampere University |

## Этапы работы

1. EDA исходных датасетов.
2. Ручное обогащение: сбор новых реальных данных с указанием источников (URL, дата обращения, лицензия).
3. Разметка собранных данных двумя аннотаторами по codebook, с замером времени на каждый объект.
4. Синтетическое обогащение: не меньше 100 % от объёма исходного датасета.
5. Оценка качества обогащения:
   - разметка: процент согласия и Cohen's κ;
   - синтетический текст: перплексия;
   - распределения: t-SNE / UMAP.
6. Обучение на четырёх вариантах датасета: без обогащения, с ручным, с ИИ-обогащением, с совместным.
   Гиперпараметры и random seed одинаковые во всех вариантах.
7. Оценка устойчивости: k-fold, разброс метрик, кривые обучения, тест на шум и искажения.
8. Отчёт по ГОСТ 7.32.

## Структура репозитория

```
docs/                       codebook разметки и сопутствующие документы
  codebook_sms_spam.md
  codebook_clotho.md
data/
  sms_spam/
    raw/                    исходный датасет UCI (в git не хранится)
    manual/                 вручную собранные сообщения + реестр источников
    synthetic/              сгенерированные сообщения
  clotho/
    raw/                    исходный Clotho (в git не хранится, ~7 ГБ)
    manual/                 вручную собранные аудио + реестр источников
    synthetic/              аугментированное аудио
labeling/
  label_studio/             конфигурации разметки для Label Studio
  annotations/              выгрузки разметки (JSON из Label Studio)
src/                        скрипты: проверка дубликатов, согласие аннотаторов и т. д.
notebooks/                  EDA, обучение, анализ
reports/                    отчёт и рисунки
```

## Ручной сбор данных

Каждый участник собирает половину: 50 spam + 50 ham и 25 аудиозаписей.
Чтобы id не пересекались, у каждого свой префикс (`m` — Максим).

### SMS Spam

| Класс | Источник | Лицензия |
|---|---|---|
| spam | [SMS Phishing Dataset](https://data.mendeley.com/datasets/f45bkkt8pr/1) (Mishra & Soni, 2022), метки spam и smishing | CC BY 4.0 |
| ham | [NUS SMS Corpus](https://github.com/kite1988/nus-sms-corpus) (Chen & Kan, 2013), английская часть | для исследований, с цитированием |

Оба корпуса скачиваются в `data/sms_spam/raw/ext_mendeley/` и `data/sms_spam/raw/ext_nus/`.
Отбор делает `src/collect_sms.py` с фиксированным seed, фильтры описаны в начале файла.
Главный фильтр: сообщения нет в исходном SMS Spam Collection — ни точной копии,
ни почти-копии, ни обрезанной версии.
Результат: `data/sms_spam/manual/messages.csv` (с меткой источника `source_label`)
и `data/sms_spam/manual/sources.csv`.

```bash
python3 src/collect_sms.py --spam 50 --ham 50 --prefix m --collected-by maxim
python3 src/make_ls_tasks.py data/sms_spam/manual/messages.csv labeling/tasks/sms_spam_tasks.json
```

Второй участник запускает сбор с другим префиксом и seed и передаёт уже собранный файл
в `--exclude`, чтобы не взять те же сообщения.

### Clotho

Звуки выбираются вручную на freesound.org.
Ссылки записываются в `data/clotho/manual/urls.txt`, там же подсказка по категориям.
Затем скрипт скачивает превью, проверяет лицензию, длительность и отсутствие в Clotho,
и заполняет `data/clotho/manual/sources.csv`:

```bash
python3 src/fetch_freesound.py --collected-by maxim --prefix m
```

Строки других участников в реестре сохраняются: скрипт заменяет только строки с тем же `--collected-by`.

Аудиофайлы в git не хранятся. Чтобы скачать все записи из реестра (например, для второй разметки),
не меняя сам реестр:

```bash
python3 src/fetch_freesound.py --download-only
```

Перед разметкой преобразуем MP3 в WAV (моно, 44 100 Гц, PCM 16 бит):

```bash
python3 -m pip install numpy scipy soundfile
python3 src/mp3_to_wav.py
```

WAV сохраняются рядом с MP3 в `data/clotho/manual/audio/`. Уже существующие WAV
скрипт пропускает. В Label Studio импортируем только `.wav`, чтобы не создавать
две задачи на одну запись. Другую папку для результатов можно указать через
`--output-dir data/clotho/manual/audio_wav`.

## Как работаем с разметкой

1. Ставим Label Studio: `pip install label-studio`, запуск — `label-studio start`.
2. Создаём проект и вставляем конфигурацию из `labeling/label_studio/`
   (Settings → Labeling Interface → Code). Задачи импортируем:
   для SMS — файл `labeling/tasks/sms_spam_tasks.json`,
   для Clotho — сами файлы из `data/clotho/manual/audio/` (Import → Upload Files).
3. Каждый аннотатор размечает **все** объекты под своим аккаунтом и не видит чужую разметку.
4. Экспортируем проект в JSON (Export → JSON) и кладём в `labeling/annotations/`
   под именем `<датасет>_<аннотатор>.json`, например `sms_maxim.json`.
5. Считаем согласие. У каждого своя установка Label Studio, поэтому передаём два файла.
   Задачи сопоставляются по `msg_id` (SMS) и по `fs_<sound_id>` в имени файла (аудио),
   так что MP3 и WAV одной записи считаются одним объектом:

```bash
python3 src/agreement.py --a labeling/annotations/sms_maxim.json --b labeling/annotations/sms_vitaliy.json \
    --choices label --out labeling/annotations/sms_merged.csv
python3 src/agreement.py --a labeling/annotations/clotho_maxim.json --b labeling/annotations/clotho_vitaliy.json \
    --choices main_source environment speech --texts caption_1 caption_2 --out labeling/annotations/clotho_merged.csv
```

6. Разбираем расхождения вдвоём и записываем решения в `labeling/annotations/sms_resolutions.csv`
   (id, итоговая метка, правило codebook, комментарий). Затем собираем итоговые метки:

```bash
python3 src/finalize_labels.py
```

Результат — `data/sms_spam/manual/messages_labeled.csv`. Скрипт также печатает, насколько итоговые
метки совпадают с меткой источника.

**Итоги разметки SMS (17.11.2026):** 100 сообщений, согласие 94 %, Cohen's κ = 0,88,
6 расхождений разобрано, codebook обновлён до версии 1.0.

Label Studio сам записывает время работы над каждым объектом (`lead_time`).
Скрипт выводит среднее, медиану и сумму по каждому аннотатору. Объекты дольше 15 минут
(вкладку забыли открытой) в среднее не входят.

Перед разметкой проверяем, что собранного нет в исходных датасетах:

```bash
python3 src/check_duplicates.py sms --original data/sms_spam/raw/SMSSpamCollection --new data/sms_spam/manual/messages.csv
python3 src/check_duplicates.py clotho --metadata-dir data/clotho/raw --new data/clotho/manual/sources.csv
```

## Используемые библиотеки

Список — в `requirements.txt`. Он будет пополняться по ходу работы.

## Журнал работ

Журнал ведётся в Google Docs, ссылка есть у участников.
