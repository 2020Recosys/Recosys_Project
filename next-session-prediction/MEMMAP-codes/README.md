# MEMMAP 관련 코드 설명
1. MEMMAP-CV-1-2-1-3.py: X, y를 만들고 이를 각각 memmap 파일로 저장 후, 10개 cv 각각에 쓸 train, test memmap file을 생성
2. MEMMAP-{DNN,DT,GNB,LR,RF,SVC,XGB}.py: 위에서 생성된 10개의 CV memmap file들을 가지고 모델 학습 및 성능 score가 담긴 pickle 파일 생성

참고: XGB의 경우 메모리 에러로 돌아가지 않아 Random Forest로 대체
