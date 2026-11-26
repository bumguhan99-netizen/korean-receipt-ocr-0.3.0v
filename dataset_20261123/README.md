# 영수증 OCR 학습 데이터셋

`ocr-receipt-dataset`의 원본 634장을 한국어 PP-OCRv5 **문자 인식** 학습용으로 가공했다. 원본 파일명과 SHA-256은 [전처리 기록](audit/preprocessing.jsonl)에 남겼고, 산출물은 번호를 붙여 통일했다.

| 분할 | 영수증 | 학습·평가 목록의 줄 | 파일 |
| --- | ---: | ---: | --- |
| 학습 | 513 | 6,492 | [train.txt](train.txt) |
| 검증 | 58 | 1,384 | [validation.txt](validation.txt) |
| 테스트 | 57 | 1,304 | [test.txt](test.txt) |
| 격리 | 6 | 0 | [분할 기록](audit/receipt_links.json) |
| 합계 | 634 | 9,180 | [무결성 보고서](audit/validation.json) |

학습 목록은 이미지 대조를 거친 3,710줄과 두 OCR 결과가 일치한 자동 라벨 2,782줄을 포함한다. [train_verified.txt](train_verified.txt)와 [train_pseudo.txt](train_pseudo.txt)에서 각각 읽을 수 있다. 검증과 테스트는 프로젝트에서 이미 이미지 대조를 완료한 줄을 사용한다. 두 세트는 이 프로젝트의 기존 평가 자료와 겹친다.

## 데이터 구성

- `images/receipt_0001.png` 형식의 RGB 영수증 634장.
- `labels/receipt_0001.json` 형식의 영수증별 라벨 634개. 문자 후보, 4점 박스, 줄 크롭 경로, 검수 상태, 보류 사유를 기록한다.
- `crops/train/receipt_0001_line_001.png` 형식의 줄 크롭 총 20,204개. 분할별 폴더에 저장했다.
- `audit/`에 원본명 매핑, 파일 해시, 분할 근거, 첫 OCR 결과, 이미지 대조와 재크롭 이력, 최종 검증을 보관했다.
- `tools/`에 이 데이터셋을 생성하고 검사한 코드를 보관했다.

## 적용한 전처리와 라벨 규칙

1. 원본의 EXIF 방향을 적용했다. 투명 픽셀은 흰색 배경에 합성하고 RGB PNG로 저장했다. 새 이미지의 긴 변이 2,400px를 넘을 때는 종횡비를 유지해 축소했다.
2. 과거 라벨 자료와 EXIF 적용 후의 RGB 픽셀 및 크기가 모두 일치하는 271장을 연결했다. 기존 라벨과 좌표가 대응하는 기울기 보정 이미지를 사용했다.
3. 픽셀 해시, 지각 해시, 연속 촬영 파일 번호로 묶음을 확인했다. 기존 분할을 유지했다. 서로 다른 기존 분할이 속한 연속 촬영 묶음 3개는 6장 모두 격리했다. 기존 라벨이 없는 363장은 학습으로 배치했다.
4. 새 영수증은 한국어 Windows OCR로 영역과 문자를 얻었다. 넓은 열 간격은 분리하고, 기울어진 영역을 4점 좌표로 잘랐다. 한국어 PP-OCRv5로 줄 크롭을 다시 읽었다. 두 모델이 공백과 기호를 포함해 일치하고, PP-OCRv5 평균 문자 신뢰도 0.93 이상·최저 문자 신뢰도 0.75 이상인 후보를 선별했다.
5. 자동 선별된 줄에서 라벨을 가린 표본 36개를 읽었다. 30개는 일치했고, 1개는 전사를 수정했으며, 5개는 잘림 또는 혼입으로 보류했다. 자동 통과 줄이 없던 44장은 재검출하거나 원본에서 영역을 직접 잡고, 읽을 수 있는 66줄을 추가했다. [직접 확인 기록](audit/visual/changes.json)
6. 텍스트를 NFC로 정규화하고 앞뒤와 중복 공백을 정리했다. v3 인코더의 여유 길이와 기존 사전에 맞춰 24자 이상과 사전 밖 문자는 학습·평가 목록에서 보류했다. 보류된 문자는 영수증별 JSON에 남겼다.
7. 이미지·크롭의 디코딩, 원본 해시, 매핑, 박스 경계, 글자 사전과 목록을 검사했다. 활성 분할 간 영수증 묶음 및 동일 크롭 픽셀 중복은 각각 0건이다.

자동 라벨 2,782줄은 문자별 전수 검수를 거치지 않았다. 두 엔진 중 하나라도 다른 결과를 냈거나 기하·사전 조건을 통과하지 못한 후보는 JSON에서 추적할 수 있다. 현재 좌표와 라벨은 인식용 줄 크롭을 위한 것이며 영수증 전체 문자 영역의 검출 정답으로 쓰기에는 커버리지가 제한적이다.

| JSON 상태 | 뜻 |
| --- | --- |
| `verified` | 과거 이미지 대조 또는 이번 직접 판독으로 전사가 확인됨 |
| `pseudo` | 학습 분할에서 두 OCR의 전사와 신뢰도 조건을 통과함 |
| `needs_review` | 판독 불일치·화질·경계 등으로 보류. OCR 결과는 후보로 기록됨 |
| `excluded` | 기존 제외 근거가 있거나 인식할 문자가 없는 영역 |

## PaddleOCR 연결

목록은 UTF-8이며 줄마다 `상대 이미지 경로 + 탭 + 전사` 형식이다. 기존 `ocr_v3/training.yaml`을 사용할 때 `Train.dataset.data_dir`와 `Eval.dataset.data_dir`를 이 폴더의 학습 환경 내 절대경로로 지정한다. `Train.dataset.label_file_list`는 `train.txt`, `Eval.dataset.label_file_list`는 `validation.txt`, `Global.character_dict_path`는 `characters.txt`의 절대경로를 사용한다. `Global.use_space_char`는 `true`를 유지한다. 최종 평가에는 `test.txt`를 사용한다.

검증을 다시 실행할 때는 이 폴더에서 다음 명령을 실행한다. `--source`를 지정하면 원본 해시도 대조한다.

```powershell
python -B tools/export_and_verify.py --check-only
python -B tools/export_and_verify.py --check-only --source ../ocr-receipt-dataset
```

## 재현 자료

Windows의 한국어 OCR, Python 3.12, Pillow 11.3, NumPy, ONNX Runtime 1.32, OpenCV 4.10을 사용했다. 새 빈 폴더에 `tools/`와 `audit/`를 준비한 뒤 `prepare.py`, `read_windows.ps1`, `build_labels.py`, `make_review.py`, `redetect.py`, `grow_review.py`, `apply_visual.py`, `export_and_verify.py` 순서로 실행한다. 직접 판독은 `audit/visual/`의 `pseudo_review_v2.json`, `direct_readings.json`, `manual_boxes.json`을 재사용한다. 모델 가중치는 [RapidOCR 모델 목록](https://rapidai.github.io/RapidOCRDocs/main/en/model_list/)과 [공식 체크섬 레지스트리](https://github.com/RapidAI/RapidOCR/blob/main/python/rapidocr/default_models.yaml)의 v3.10.0 ONNX 파일을 사용했고, 스크립트에서 SHA-256을 확인한다.
