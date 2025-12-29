import bnlearn as bn
import os

# 1. 저장할 디렉터리 생성
output_dir = 'data/raw/bnlearn'
os.makedirs(output_dir, exist_ok=True)
print(f"모든 파일은 '{output_dir}/' 폴더에 저장됩니다.\n")

# 2. bnlearn에 포함된 예제 데이터셋 목록 (하드코딩)
dataset_list = ['sprinkler', 'alarm', 'andes', 'asia', 'sachs', 'water']
print(f"총 {len(dataset_list)}개의 데이터셋을 처리합니다.")
print(dataset_list)
print("-" * 30)

# 3. 각 데이터셋을 순회하며 데이터와 그래프(모델) 저장
for i, name in enumerate(dataset_list):
    print(f"[{i+1}/{len(dataset_list)}] '{name}' 데이터셋 처리 중...")
    
    try:
        # 각 데이터셋별 디렉터리 생성
        dataset_dir = os.path.join(output_dir, name)
        os.makedirs(dataset_dir, exist_ok=True)
        
        # (1) 데이터셋 불러오기
        df = bn.import_example(data=name)
        
        # (2) 원본 데이터를 CSV 파일로 저장
        csv_path = os.path.join(dataset_dir, 'data.csv')
        df.to_csv(csv_path, index=False)
        
        # (3) 데이터로부터 그래프 구조 학습 (Hill-Climbing 방식 사용)
        # 'methodtype='hc''는 가장 일반적인 구조 학습 알고리즘 중 하나입니다.
        model = bn.structure_learning.fit(df, methodtype='hc')
        
        # (4) 학습된 모델(그래프 구조 + 파라미터)을 pickle 파일로 저장
        model_path = os.path.join(dataset_dir, 'model')
        bn.save(model, filepath=model_path, overwrite=True)
        
        # (5) 그래프 시각화(PNG 이미지) 저장
        # bn.plot_graphviz는 graphviz 객체를 반환합니다.
        G = bn.plot_graphviz(model)
        graph_path = os.path.join(dataset_dir, 'graph')
        # .render() 함수로 파일 저장 (format='png', view=False로 창이 안 뜨게 함)
        G.render(filename=graph_path, format='png', view=False, cleanup=True)
        
        print(f"  -> '{name}' 처리 완료: CSV, Model, PNG 이미지 저장 성공")

    except Exception as e:
        # 일부 데이터셋은 전처리가 필요하거나 특정 파라미터가 필요할 수 있습니다.
        print(f"  -> ERROR: '{name}' 처리 중 오류 발생: {e}")

print("-" * 30)
print(f"모든 작업이 완료되었습니다. '{output_dir}' 폴더를 확인하세요.")