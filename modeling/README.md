#### crawling.py

- 희망하는 검색어 관련 네이버 카페, 블로그 게시글 크롤링

```
python crawling.py --cafe true --blog true
```

- `--cafe` : 네이버 카페 크롤링 여부 - `true` or `false`

- `--blog` : 네이버 블로그 크롤링 여부 - `true` or `false`

- 실행 시, 검색어 입력 받고 바로 크롤링 실행

### preprocessing.py

- 데이터셋 생성

```python
python preprocessing.py --prod path --serv path --cust path --prod_info path --scaler path
```

- `--prod`: Demo 데이터 경로

- `--serv`: 상품 구매 정보 데이터 경로

- `--cust`: 제휴사 이용 정보 데이터 경로

- `--prod_info`: 상품 분류 정보 데이터 경로

- `--scaler`: RobustScaler 경로 - `''` 입력 시, 학습 / 경로 입력 시 불러와서 변환만 수행

## modeling

### main.py

- 학습, 평가 실행

```
python main.py train_config.json
```

### inference.py

- 추론 시행

```
python inference.py inference_config.json
```


