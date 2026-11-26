# OCR 학습
자체 영수증 원본은 data/originals/phone_receipts와 data/originals/additional_receipts, KORIE 전체 영수증과 영역 좌표는 data/korie/receipts에 있다. data/korie/text_crops는 공식 OCR 배포본이고 data/korie/receipt_fields는 정보추출 자료다.

원본 영수증 단위로 분할한 뒤 새 크롭을 data/text_crops에 저장한다. data/labels/train.txt와 validation.txt에는 각 줄에 프로젝트 기준 상대 경로와 정답 문자열을 탭으로 구분하여 쓴다. 테스트 영수증은 최종 평가용으로 보관한다.

모델은 korean_PP-OCRv5_mobile_rec 공식 사전학습 가중치에서 시작한다. training.yaml에는 초기 설정으로 30 epoch, 배치 32, 학습률 0.0001을 지정했다. 크롭과 라벨을 준비한 뒤 이 폴더에서 docker compose run --rm train을 실행한다.
